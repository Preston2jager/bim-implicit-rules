# --- repo-root bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path.cwd()
if not (_ROOT / "src" / "Graphormer.py").exists():
    raise SystemExit("run from the SDF_3 repository root (cd ~/Projects/SDF_3)")
# --- end bootstrap ---
"""Plan 2 / P2-3 spot check of generated graphs (exit 1 on any failure, so the queue stops before training).

For every graph of 3 copies (2 train + 1 holdout) x all 4 room states x a few wall combos:
  * node count 38-40
  * space names: exactly 1 Toilet, 1 Corridor, 1 LivingRoom, 1 Bedroom, 3 Store
  * TARGET_DOOR_SYNTH links only to spaces named Corridor/LivingRoom/Bedroom (102/103/106)
  * TARGET_DOOR_SYNTH wall_s in {+-0.495, +-1.485, +-2.475}
  * world position of the synth door = host_wall_center + wall_s * host_wall_axis
  * across the 4 room states of one copy+combo the Toilet node sits at 4 different spaces
Also: every slot value occurs over the whole holdout set. Writes $SDF3_RESULTS/P2_split/graph_check.json.

    python scripts/v2/p2_graph_check.py --run V2_P2_inductive
"""
import argparse, json, os, random, re
from collections import Counter, defaultdict
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); args = ap.parse_args()
out_dir = _ROOT / "data/runs" / args.run / "output"
REP = _Path(os.environ.get("SDF3_RESULTS", "data/results")) / "P2_split"
PAT = re.compile(r"graph_room(\d)_aug(\d+)_(L\w+?)_(R\w+?)_W(\d)_graph\.json")
SLOTS = np.array([-2.475, -1.485, -0.495, 0.495, 1.485, 2.475])
errs, n_checked, node_counts, slot_hits = [], 0, Counter(), Counter()


def check(f):
    global n_checked
    g = json.loads(f.read_text())["graphs"][0]
    nodes = {n["id"]: n for n in g["nodes"]}
    n_checked += 1
    node_counts[len(nodes)] += 1
    if not 38 <= len(nodes) <= 40:
        errs.append(f"{f.name}: {len(nodes)} nodes")
    names = Counter(n["Name"] for n in nodes.values() if n["label"] == "IfcSpace")
    want = Counter({"Toilet": 1, "Corridor": 1, "LivingRoom": 1, "Bedroom": 1, "Store": 3})
    if names != want:
        errs.append(f"{f.name}: space names {dict(names)}")
    sd = nodes.get("TARGET_DOOR_SYNTH")
    if sd is None:
        errs.append(f"{f.name}: no synth door"); return None
    nb = {l["end"] if l["start"] == "TARGET_DOOR_SYNTH" else l["start"] for l in g["links"]
          if "TARGET_DOOR_SYNTH" in (l["start"], l["end"])}
    nb_sp = {nodes[i]["Name"] for i in nb if i in nodes and nodes[i]["label"] == "IfcSpace"}
    if not nb_sp or not nb_sp <= {"Corridor", "LivingRoom", "Bedroom"}:
        errs.append(f"{f.name}: synth door spaces {nb_sp}")
    s = sd["wall_s"]
    k = int(np.argmin(np.abs(SLOTS - s)))
    if abs(SLOTS[k] - s) > 0.005:
        errs.append(f"{f.name}: wall_s {s:.4f} not a slot")
    slot_hits[k] += 1
    p = np.array(sd["host_wall_center"]) + s * np.array(sd["host_wall_axis"])
    if np.linalg.norm(p - np.array(sd["coordinates"])) > 0.01:
        errs.append(f"{f.name}: synth door world pos off by {np.linalg.norm(p - np.array(sd['coordinates'])):.3f} m")
    toilet = [i for i, n in nodes.items() if n["Name"] == "Toilet"]
    return toilet[0] if toilet else None


train = sorted(out_dir.glob("graph_room*.json")); hold = sorted((out_dir / "holdout").glob("graph_room*.json"))
by_key = defaultdict(dict)
for f in train + hold:
    m = PAT.match(f.name)
    by_key[(f.parent.name, m.group(2))].setdefault((m.group(3), m.group(4), m.group(5)), {})[m.group(1)] = f
random.seed(0)
keys_tr = sorted({k for k in by_key if k[0] != "holdout"}); keys_ho = sorted({k for k in by_key if k[0] == "holdout"})
picked = random.sample(keys_tr, 2) + random.sample(keys_ho, 1)
for key in picked:
    combos = sorted(by_key[key])
    for combo in random.sample(combos, min(12, len(combos))):
        toilets = {r: check(f) for r, f in by_key[key][combo].items()}
        if len(toilets) == 4 and len(set(toilets.values())) != 4:
            errs.append(f"{key} {combo}: Toilet not on 4 distinct spaces across room states")
# every slot used somewhere in the holdout
all_s = Counter()
for f in hold:
    g = json.loads(f.read_text())["graphs"][0]
    s = next(n["wall_s"] for n in g["nodes"] if n["id"] == "TARGET_DOOR_SYNTH")
    all_s[int(np.argmin(np.abs(SLOTS - s)))] += 1
if len(all_s) != 6:
    errs.append(f"holdout uses only slots {sorted(all_s)}")
rep = {"run": args.run, "copies_checked": [list(k) for k in picked], "graphs_checked": n_checked,
       "node_counts": dict(node_counts), "holdout_slot_counts": {str(SLOTS[k]): v for k, v in sorted(all_s.items())},
       "n_train_graphs": len(train), "n_holdout_graphs": len(hold), "errors": errs[:50], "n_errors": len(errs)}
REP.mkdir(parents=True, exist_ok=True)
(REP / f"graph_check_{args.run}.json").write_text(json.dumps(rep, indent=1))
print(json.dumps({k: v for k, v in rep.items() if k != "errors"}, indent=1))
for e in errs[:20]:
    print("FAIL", e)
print("GRAPH CHECK", "PASSED" if not errs else "FAILED")
if not errs:  # pass-only marker = the queue output (exit codes alone are not trusted)
    (REP / f"graph_check_{args.run}.PASSED").write_text("ok")
raise SystemExit(1 if errs else 0)
