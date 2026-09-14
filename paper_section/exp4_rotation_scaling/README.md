# 实验四：旋转多样性 scaling（copies 81→161）

**设定**：固定 scalar_s 头 + pose_distill_weight 0.3 + 去身份化 Name，
仅把刚体变换 copy 数从 81（6 holdout）翻倍到 161（12 holdout）：

- 81 copies：run `2026JUL112325`，archive `scalar_s_pose_w0.3`
- 161 copies：run `2026JUL122211`（第八轮，全程约 15h：hash 嵌入早停 +
  Graphormer 早停 45/100ep，Val MSE 3.5e-5；12 万图触发 host→device 流式训练）

**结论**：
1. holdout 整体侧别 88.2→92.7%，>0.5 m 错误占比 19.3→10.8%，
   per-copy 最差 72→81%；12 个 holdout 变换中 3 个完美。
   翻倍旋转多样性有效但收益递减。
2. seen test 两轮都在 ~4–16 mm / 100%——所有差异都在未见变换上。
3. per-copy 方差仍是主要不确定性来源（12 点分布见图），这也是把 holdout
   变换数保持与 copies 成比例（3/40→6/80→12/160）的原因。

**文件**：
- `extract_data.py` → `data/scaling.json`（含 per-copy 侧别准确率）
- `make_figures.py` → `figures/fig_scaling.{png,pdf}`
