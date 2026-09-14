# --- repo-root bootstrap (added when this entry point moved into scripts/) ---
# Every data path in this file is relative to the repository root, so run it from there:
#     cd ~/Projects/SDF_3 && python scripts/generate_graph_dataset.py
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# --- end bootstrap ---

import os
import sys
import json
import shutil
import hashlib
import numpy as np
from pathlib import Path
from loguru import logger
from flax.training import checkpoints
import ifcopenshell
import ifcopenshell.geom

# Target elements GUIDs
LEFT_DOOR_GUID = "0XVTLR_9H3Ox9MAKGyWioT"
RIGHT_DOOR_GUID = "0i$VJnJOTDM92a$t1VH2bb"
LEFT_WALL_GUID = "02414yPEv25fF9z0uqxlni"
RIGHT_WALL_GUID = "3U2rKl8q5DVfhs17a5SE44"

# Space node GUIDs
ROOM_100_GUID = "1JL9rqIu1D$A7C0Nubeag_"
ROOM_101_GUID = "1JL9rqIu1D$A7C0Nubeagz"
ROOM_102_GUID = "1JL9rqIu1D$A7C0Nubeagm"
ROOM_103_GUID = "1JL9rqIu1D$A7C0Nubeagt"
ROOM_104_GUID = "1JL9rqIu1D$A7C0NubeahA"
ROOM_105_GUID = "1JL9rqIu1D$A7C0Nubeah9"

def generate_stable_latent(guid: str) -> list[float]:
    """Generates a stable 128-dimensional latent vector seeded by element GUID."""
    md5_hash = hashlib.md5(str(guid).encode('utf-8')).hexdigest()
    seed = int(md5_hash[:8], 16) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    return rng.random(128, dtype=np.float32).tolist()

def determine_doors_state(room_state: int, has_left: bool, len_left: int, has_right: bool, len_right: int) -> tuple[bool, bool]:
    """
    Implements the target door deletion/retention logic according to guid.md.
    Returns (keep_left_door, keep_right_door).
    """
    toilet_in_100_101 = (room_state in (1, 2))
    is_100_toilet = (room_state == 1)
    is_104_toilet = (room_state == 3)
    
    if toilet_in_100_101:
        # Rule 1: changing wall left only > 1m, or both but left > 1m and right < 1m
        if (has_left and len_left > 1000 and not has_right) or (has_left and len_left > 1000 and has_right and len_right < 1000):
            return True, False
        # Rule 2: changing wall right only > 1m, or both but left < 1m and right > 1m
        elif (has_right and len_right > 1000 and not has_left) or (has_right and len_right > 1000 and has_left and len_left < 1000):
            return False, True
        # Rule 3: both > 1m
        elif has_left and len_left > 1000 and has_right and len_right > 1000:
            if is_100_toilet:
                return True, False
            else:
                return False, True
        # Rule 4: single side < 1m, or both < 1m, or neither
        else:
            if is_100_toilet:
                return True, False
            else:
                return False, True
    else:
        # Rule 1: changing wall left only > 1m, or both but left > 1m and right < 1m
        if (has_left and len_left > 1000 and not has_right) or (has_left and len_left > 1000 and has_right and len_right < 1000):
            return True, False
        # Rule 2: changing wall right only > 1m, or both but left < 1m and right > 1m
        elif (has_right and len_right > 1000 and not has_left) or (has_right and len_right > 1000 and has_left and len_left < 1000):
            return False, True
        # Rule 3: both > 1m, or single < 1m, or both < 1m, or neither
        else:
            if is_104_toilet:
                return True, False
            else:
                return False, True

def main():
    # Run folder comes from argv; default is the most recently modified run.
    # (This used to hardcode data/runs/2026JUN301147, which no longer exists.)
    if len(sys.argv) > 1:
        work_folder = Path(sys.argv[1]).resolve()
    else:
        runs_root = Path("data/runs")
        candidates = sorted(
            (d for d in runs_root.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        ) if runs_root.is_dir() else []
        if not candidates:
            logger.error(
                "No run folder under data/runs. Pass one explicitly: "
                "python scripts/generate_graph_dataset.py data/runs/<RUN>"
            )
            sys.exit(1)
        work_folder = candidates[0].resolve()
    if not work_folder.exists():
        logger.error(f"Work folder {work_folder} does not exist.")
        sys.exit(1)
        
    logger.info(f"Using work folder: {work_folder}")
    
    # 1. Restore trained embeddings
    checkpoint_dir = work_folder / "checkpoints"
    logger.info(f"Restoring trained embeddings from checkpoint folder: {checkpoint_dir}")
    best_state = checkpoints.restore_checkpoint(ckpt_dir=str(checkpoint_dir), target=None, prefix="best_")
    if best_state is None:
        best_state = checkpoints.restore_checkpoint(ckpt_dir=str(checkpoint_dir), target=None)
        
    if best_state is None:
        logger.error("Failed to restore flax checkpoint containing latent embeddings.")
        sys.exit(1)
        
    embeddings = np.array(best_state['params']['latent_embedding']['embedding'])
    logger.success(f"Loaded latent embeddings of shape: {embeddings.shape}")
    
    # Load idx_str2int_dict to map elements to embeddings
    idx_str2int_path = work_folder / "intermediate" / "idx_str2int_dict.npy"
    if not idx_str2int_path.exists():
        logger.error(f"idx_str2int_dict.npy not found in {idx_str2int_path.parent}")
        sys.exit(1)
        
    idx_str2int = np.load(idx_str2int_path, allow_pickle=True).item()
    
    # Extract the date prefix dynamically
    any_key = next(iter(idx_str2int.keys()))
    prefix_date = any_key.split('_')[0]
    logger.info(f"Detected prefix date for embeddings: {prefix_date}")
    
    # 2. Extract base coordinates and transformation matrices for left/right target walls from IFC
    base_ifc_path = Path("data/input/input_ifcs/Data_base_1.ifc")
    if not base_ifc_path.exists():
        base_ifc_path = work_folder / "input" / "input_ifcs" / "Data_base_1.ifc"
        
    if not base_ifc_path.exists():
        logger.error("Data_base_1.ifc not found.")
        sys.exit(1)
        
    logger.info(f"Reading base wall geometries from: {base_ifc_path}")
    model = ifcopenshell.open(str(base_ifc_path))
    
    settings_local = ifcopenshell.geom.settings()
    settings_local.set(settings_local.USE_WORLD_COORDS, False)
    
    # Left Wall geometry extraction
    left_wall = model.by_guid(LEFT_WALL_GUID)
    left_wall_shape = ifcopenshell.geom.create_shape(settings_local, left_wall)
    T_left = np.asarray(left_wall_shape.transformation.matrix, dtype=float).reshape(4, 4).T
    
    # Right Wall geometry extraction
    right_wall = model.by_guid(RIGHT_WALL_GUID)
    right_wall_shape = ifcopenshell.geom.create_shape(settings_local, right_wall)
    T_right = np.asarray(right_wall_shape.transformation.matrix, dtype=float).reshape(4, 4).T
    
    # 3. Create the 4 renamed IFC files
    expanded_folder = work_folder / "intermediate" / "expanded_ifcs"
    expanded_folder.mkdir(parents=True, exist_ok=True)
    
    room_states_config = {
        1: {
            ROOM_100_GUID: ("Toilet", "Toilet"),
            ROOM_101_GUID: ("Store", "Store"),
            ROOM_102_GUID: ("Bedroom", "Bedroom"),
            ROOM_103_GUID: ("LivingRoom", "Living Room"),
            ROOM_104_GUID: ("Store", "Store"),
            ROOM_105_GUID: ("Store", "Store"),
        },
        2: {
            ROOM_100_GUID: ("Store", "Store"),
            ROOM_101_GUID: ("Toilet", "Toilet"),
            ROOM_102_GUID: ("Bedroom", "Bedroom"),
            ROOM_103_GUID: ("LivingRoom", "Living Room"),
            ROOM_104_GUID: ("Store", "Store"),
            ROOM_105_GUID: ("Store", "Store"),
        },
        3: {
            ROOM_100_GUID: ("Store", "Store"),
            ROOM_101_GUID: ("Store", "Store"),
            ROOM_102_GUID: ("Bedroom", "Bedroom"),
            ROOM_103_GUID: ("LivingRoom", "Living Room"),
            ROOM_104_GUID: ("Toilet", "Toilet"),
            ROOM_105_GUID: ("Store", "Store"),
        },
        4: {
            ROOM_100_GUID: ("Store", "Store"),
            ROOM_101_GUID: ("Store", "Store"),
            ROOM_102_GUID: ("Bedroom", "Bedroom"),
            ROOM_103_GUID: ("LivingRoom", "Living Room"),
            ROOM_104_GUID: ("Store", "Store"),
            ROOM_105_GUID: ("Toilet", "Toilet"),
        }
    }
    
    logger.info("Generating 4 renamed IFC files...")
    for s_idx, space_names in room_states_config.items():
        ifc_copy = ifcopenshell.open(str(base_ifc_path))
        for guid, (name, long_name) in space_names.items():
            space = ifc_copy.by_guid(guid)
            if space:
                space.Name = name
                space.LongName = long_name
        out_path = expanded_folder / f"Room_State_{s_idx}.ifc"
        ifc_copy.write(str(out_path))
        logger.info(f"  + Saved {out_path.name}")
        
    # 4. Load the base graph template JSON
    base_graph_path = work_folder / "output" / "Data_base_1_graph.json"
    if not base_graph_path.exists():
        logger.error(f"Base template graph not found at {base_graph_path}. Make sure to generate it first.")
        sys.exit(1)
        
    with open(base_graph_path, "r", encoding="utf-8") as f:
        base_graph_data = json.load(f)
        
    template_nodes = base_graph_data["graphs"][0]["nodes"]
    template_links = base_graph_data["graphs"][0]["links"]
    logger.info(f"Loaded template graph with {len(template_nodes)} nodes and {len(template_links)} links.")
    
    # 5. Define possible wall states: Absent, or Present with length L in range(100, 3100, 100)
    wall_lengths = list(range(100, 3100, 100))
    wall_options = [None] + wall_lengths # None means Absent
    
    output_folder = work_folder / "output"
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Clear output folder of other graph files to avoid pollution
    for p in output_folder.glob("graph_room*.json"):
        p.unlink()
        
    total_graphs_generated = 0
    
    logger.info("Starting GNN Graph Dataset generation loop...")
    
    for room_state in (1, 2, 3, 4):
        space_config = room_states_config[room_state]
        
        for len_left in wall_options:
            has_left = (len_left is not None)
            for len_right in wall_options:
                has_right = (len_right is not None)
                
                # Determine target doors keep/delete states
                keep_left_door, keep_right_door = determine_doors_state(
                    room_state, has_left, len_left or 0, has_right, len_right or 0
                )
                
                # Filter nodes
                new_nodes = []
                deleted_node_guids = set()
                if not has_left:
                    deleted_node_guids.add(LEFT_WALL_GUID)
                if not has_right:
                    deleted_node_guids.add(RIGHT_WALL_GUID)
                if not keep_left_door:
                    deleted_node_guids.add(LEFT_DOOR_GUID)
                if not keep_right_door:
                    deleted_node_guids.add(RIGHT_DOOR_GUID)
                    
                for t_node in template_nodes:
                    node_id = t_node["id"]
                    if node_id in deleted_node_guids:
                        continue
                        
                    # Copy node
                    node = dict(t_node)
                    
                    # Update space names/types
                    if node_id in space_config:
                        node["Name"] = space_config[node_id][0]
                        
                    # Update wall geometries and coordinates
                    if node_id == LEFT_WALL_GUID and has_left:
                        # Set coordinates dynamically using transform matrix
                        new_max_x = len_left / 1000.0
                        P_local = np.array([new_max_x / 2.0, 0.0, 0.0, 1.0])
                        P_world = T_left @ P_local
                        node["coordinates"] = [float(P_world[0]), float(P_world[1])]
                        
                        # Set trained embedding geometry
                        key = f"{prefix_date}_1_IfcWall_{LEFT_WALL_GUID}_L{len_left}"
                        if key in idx_str2int:
                            node["geometry"] = embeddings[idx_str2int[key]].tolist()
                        else:
                            node["geometry"] = generate_stable_latent(LEFT_WALL_GUID)
                            
                    elif node_id == RIGHT_WALL_GUID and has_right:
                        # Set coordinates dynamically using transform matrix
                        new_max_x = len_right / 1000.0
                        P_local = np.array([new_max_x / 2.0, 0.0, 0.0, 1.0])
                        P_world = T_right @ P_local
                        node["coordinates"] = [float(P_world[0]), float(P_world[1])]
                        
                        # Set trained embedding geometry
                        key = f"{prefix_date}_1_IfcWall_{RIGHT_WALL_GUID}_L{len_right}"
                        if key in idx_str2int:
                            node["geometry"] = embeddings[idx_str2int[key]].tolist()
                        else:
                            node["geometry"] = generate_stable_latent(RIGHT_WALL_GUID)
                            
                    # Target doors have geometry zeroed out to prevent self-leakage
                    elif node_id in (LEFT_DOOR_GUID, RIGHT_DOOR_GUID):
                        node["geometry"] = [0.0] * 128
                        
                    # Update normal nodes geometry from checkpoints
                    else:
                        key = f"{prefix_date}_1_{node['label']}_{node_id}"
                        if key in idx_str2int:
                            node["geometry"] = embeddings[idx_str2int[key]].tolist()
                        else:
                            node["geometry"] = generate_stable_latent(node_id)
                            
                    new_nodes.append(node)
                    
                # Filter links
                new_links = []
                for link in template_links:
                    start_id = link.get("start")
                    end_id = link.get("end")
                    if start_id in deleted_node_guids or end_id in deleted_node_guids:
                        continue
                    new_links.append(link)
                    
                # Construct graph dictionary
                graph_dict = {
                    "graphs": [
                        {
                            "nodes": new_nodes,
                            "links": new_links
                        }
                    ]
                }
                
                # Write to file
                l_str = f"L{len_left}" if has_left else "LAbsent"
                r_str = f"R{len_right}" if has_right else "RAbsent"
                filename = f"graph_room{room_state}_{l_str}_{r_str}_graph.json"
                out_json_path = output_folder / filename
                
                with open(out_json_path, "w", encoding="utf-8") as out_f:
                    json.dump(graph_dict, out_f, ensure_ascii=False, indent=2)
                    
                total_graphs_generated += 1
                
    logger.success(f"Graph dataset generation completed! Generated {total_graphs_generated} graph files in {output_folder.as_posix()}")

if __name__ == "__main__":
    main()
