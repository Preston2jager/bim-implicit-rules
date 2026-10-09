# 实验十一：第二种户型拓扑（Data_2.ifc）

**目的**：回应审稿意见"只在一种户型上验证"。在另一套更小、拓扑不同的户型上（7 个房间，`data/input/plan2/Data_2.ifc`），用和最终模型完全相同的流程与超参数重做核心实验。

**流程**：归纳式 v2 流程，脚本在 `scripts/v2/`。
- SDF decoder 只用训练副本训练；holdout 副本的 latent 在 decoder 冻结后单独拟合，并过一道质量门（holdout 拟合 MAE / 训练 MAE ≤ 1.5）。
- 最终模型（Graphormer、scalar_s 头、蒸馏 w = 0.3）的所有评估都在训练它的那个进程里完成。
- 户型 2 的门位是 ±0.495 / ±1.485 / ±2.475 m（间距 0.99 m），户型 1 是 0.95 m。

## 结论

侧别准确率按 12 个 holdout 副本做 cluster bootstrap，括号里是 95% CI。

| | 户型 1 | 户型 2（81 份副本） | 户型 2（161 份副本） |
|---|---|---|---|
| 侧别，seed 42 | 80.8% | 67.0% [61.2, 72.4] | 64.4% [58.6, 69.9] |
| 侧别，多 seed | 79.1 ± 1.8（5 seed） | 68.2 ± 3.6（3 seed） | — |
| window / wall / semantic，seed 42 | 71.2 / 80.8 / 99.7 | 69.8 / 59.9 / 69.6 | 58.2 / 68.2 / 72.4 |
| 蒸馏增益（与 w = 0 配对检验） | +7.8 pp，p = 0.001 | +7.8 pp（三 seed 平均），p = 0.0098 | — |
| SDF 质量门比值 | 0.60 | 0.61 | 0.58 |
| 户型 1 模型直接测户型 2（零样本） | — | **74.0%** [71.5, 76.9]，semantic 100% | — |

1. **方法能直接迁移。** 代码和超参数都不改，SDF 质量与户型 1 相同，三个 seed 都明显高于 50% 随机水平，已见副本上都是 100%。
2. **蒸馏的增益复现了。** 三 seed 平均比 w = 0 高 +7.8 pp（CI [3.1, 12.6]，12 份副本中 10 份更好），与户型 1 的 +7.8 pp 一致。
3. **准确率比户型 1 低约 11 pp，原因不是副本数。** 副本加倍到与户型 1 同规模（161 份）后是 64.4%，没有提升。
4. **户型 2 的规则学得会。** 户型 1 训练的模型没见过户型 2，直接测反而有 74.0%，semantic 100%，比任何一个用户型 2 训练的模型都好。所以差距出在"用户型 2 的数据训练"这一步：模型学到的东西泛化不到新的旋转，semantic 规则最明显（seed 42 在 Toilet 位于东侧的两种状态下只有 46–47%）。机制没有验证。

## 文件

- `data/plan2.json`：所有汇总数字（`summary`），以及 7 个 arm 在每个 holdout 副本上的侧别、准确槽位和误差 > 0.5 m 的比例（`per_copy`）。所有 CI 和配对检验都可以只用这个文件重算。
- `data/transforms_81.json`、`data/transforms_161.json`：每份副本的旋转角和平移，以及哪些是 holdout、每个 holdout 到最近训练副本的角距离。
- `make_figures.py`：只读 `data/plan2.json`，画 `figures/fig_plan2.{png,pdf}`（各 arm 的每副本侧别准确率），并重算 w = 0 的配对检验，与存档值对照打印。
- `extract_data.py`：从结果目录生成 `data/plan2.json`，需要完整跑过下面的流程。

不需要任何 run 存档，就可以重画图并核对统计：

```bash
python paper_section/exp11_second_topology/make_figures.py
```

## 从头复现

所有命令都在仓库根目录运行。结果写到 `$SDF3_RESULTS`（默认 `data/results`），中间数据写到 `data/runs/`，这两处都不进 git。硬件和户型 1 相同（16 GB 显存），总墙钟约 43 h。

```bash
PY=.venv/bin/python

# 1. 副本：80 份刚体变换 + 基准，按 sorted(*.ifc) 最后 12 份为 holdout
$PY scripts/v2/p2_src.py --name P2_src --copies 80
$PY scripts/v2/p2_transforms.py data/runs/P2_src transforms.json

# 2. SDF：采样 -> 只用训练副本训练 decoder -> 冻结 decoder 拟合 holdout，过质量门
OBJECTS_PER_COPY=60 $PY scripts/v2/v2_prepare.py --src P2_src --run V2_P2_inductive --hold-run V2_P2_holdfit
$PY scripts/v2/v2_sdf_train.py --run V2_P2_inductive
$PY scripts/v2/v2_fit_holdout.py --run V2_P2_inductive --hold-run V2_P2_holdfit --report-dir ${SDF3_RESULTS:-data/results}/V2_P2_fit

# 3. 组合图（55,476 训练 / 9,648 holdout）和自动抽查
$PY scripts/v2/v2_finalize.py --src P2_src --run V2_P2_inductive --hold-run V2_P2_holdfit --expect-train 55476 --expect-holdout 9648
$PY scripts/v2/p2_graph_check.py --run V2_P2_inductive

# 4. 最终模型三个 seed + 不蒸馏对照；每个 arm 训练后立即做分析
for s in 42 1 2; do
  $PY scripts/v2/train_arm.py --src V2_P2_inductive --tag P2_s$s --seed $s --set training.graphormer.max_device_data_gb=4
  $PY scripts/v2/t10_t08_analysis.py --run REV_P2_s$s --report data/runs/REV_P2_s$s/output/experiments/P2_s$s/holdout_evaluation_report.json --tag P2_s$s --slot-spacing 0.99
done
$PY scripts/v2/train_arm.py --src V2_P2_inductive --tag P2_w0_s42 --seed 42 --set training.graphormer.pose_distill_weight=0 --set training.graphormer.max_device_data_gb=4
$PY scripts/v2/t10_t08_analysis.py --run REV_P2_w0_s42 --report data/runs/REV_P2_w0_s42/output/experiments/P2_w0_s42/holdout_evaluation_report.json --tag P2_w0_s42 --slot-spacing 0.99
$PY scripts/v2/p2_model_tests.py
```

**跨户型零样本。** 这一步需要户型 1 的 v2 run `V2_inductive`。先用 `scripts/main.py` 的 5.2 生成户型 1 的 161 份副本 run（下面记为 `2026JUL122211`），再：

```bash
$PY scripts/v2/v2_prepare.py --src 2026JUL122211 --run V2_inductive --hold-run V2_holdfit
$PY scripts/v2/v2_sdf_train.py --run V2_inductive
$PY scripts/v2/v2_fit_holdout.py --run V2_inductive --hold-run V2_holdfit
$PY scripts/v2/v2_finalize.py --src 2026JUL122211 --run V2_inductive --hold-run V2_holdfit
```

然后：

```bash
$PY scripts/v2/p2x1_prepare.py --src-hold V2_P2_holdfit --dst V2_P2_holdfit_x1
$PY scripts/v2/v2_fit_holdout.py --run V2_inductive --hold-run V2_P2_holdfit_x1 --report-dir ${SDF3_RESULTS:-data/results}/V2_P2x1_fit
$PY scripts/v2/p2x1_graphs.py --hold-run V2_P2_holdfit_x1 --extra-root V2_P2x1_extra
$PY scripts/v2/train_arm.py --src V2_inductive --tag P1_s42_x2 --seed 42 --extra-holdout-root data/runs/V2_P2x1_extra
$PY scripts/v2/t10_t08_analysis.py --run V2_inductive --report ${SDF3_RESULTS:-data/results}/P1_s42_x2/holdout_evaluation_report.json.gz --tag P1_s42_x2
$PY scripts/v2/t10_t08_analysis.py --run V2_P2x1_extra --holdout-dir data/runs/V2_P2x1_extra/plan2_holdout \
    --report ${SDF3_RESULTS:-data/results}/P1_s42_x2/extra_plan2_holdout.json.gz --tag P1_s42_x2_on_plan2 --slot-spacing 0.99
```

**161 份副本对照。** 步骤 1–4 换成 `--name P2_src161 --copies 160`，run 名 `V2_P2_161_inductive` / `V2_P2_161_holdfit`，tag 为 `P2_161_s42`，图数是 119,796 / 9,648，只跑 seed 42。

最后汇总并导出：

```bash
$PY scripts/v2/p2_summary.py
$PY paper_section/exp11_second_topology/extract_data.py
```

## 复现时要注意

- **副本的随机变换没有固定种子。** 重新生成会得到另一组旋转和平移，所以数字会在 seed 间和副本间的波动范围内变化（户型 2 三 seed 的 SD 3.6 pp，副本间约 10 pp），不会逐位一致。原始的变换记录在 `data/transforms_*.json`。
- **户型 2 的 arm 必须加 `max_device_data_gb=4`。** 户型 2 的 padded 数据集约 8.9 GB，低于默认阈值 10 GB，会被整块放进显存，在 16 GB 卡上第 3 个 epoch 就会显存不足（OOM）。设成 4 后改为按 batch 从内存流式送入，和户型 1 的做法相同，batch 内容不变。
- **不要在新进程里重新加载 checkpoint 做评估。** 所有 holdout 和 extra holdout 都在 `train_arm.py` 训练的那个进程里评估；跨户型测试通过 `--extra-holdout-root` 完成。
- **户型 1 的词表里没有户型 2 的一些标签。** 户型 2 的 Corridor 标签和新墙、新门编码为 0，零样本结果包含这一影响。
- **P2-6 的 12 份副本不全在 holdout 目录。** `generate_combinatorial_graphs` 总会留至少一份做训练，所以 11 份进 `output/holdout`、1 份进 `output/`；`p2x1_graphs.py` 把两者合并为一个 9,648 张的 extra holdout。
- **探针上限（derivability probe）没有做**，因为该脚本写死了户型 1 的 IFC Tag 和规则引擎。
