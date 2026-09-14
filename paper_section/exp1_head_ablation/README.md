# 实验一：预测头消融（坐标系主张的证伪）

**设定**：旧版数据（4 房型 × 10 刚体变换 copy × 31×31 墙长组合，门位置二值 ±2.375 m），
run `2026JUL100130`，同一批嵌入与图上仅更换预测头重训（菜单 6）。

**三种头**：

| 头 | 组合方式 | 外部提供 | 理论不变性 |
|---|---|---|---|
| absolute | world = pred | 无 | 无 |
| offset | world = wall_center + pred | 墙心 | 平移 |
| scalar_s（axis-product 版） | world = wc + (pred·axis)·axis | 墙心+轴 | 平移（旋转仍需模型读出） |

**结论**：三种头 seen test 均 <0.09 m，但 unseen-transform holdout 全部崩溃
（4.04 / 3.43 / 3.86 m，44–82 倍差距）——模型记忆了有限的训练变换，
**没有学到可泛化的坐标系**。scalar_s 的 axis-product 版暴露 product trap：
目标 = s·axis(copy) 把变换不变的答案与 copy 旋转耦合，holdout 预测与真值精确
反相关（corr −0.999）且幅度塌缩（1.49 vs 2.375）。修正投影为 world = wc + pred_x·axis
后模型输出直接就是沿墙标量 s，完全变换不变——后续所有实验使用该版本。

**文件**：
- `extract_data.py` → `data/summary.json`（从 `experiments/exp1_absolute`、
  `experiments/exp2_offset`、`data/runs/2026JUL100130` 的评估报告提取）
- `make_figures.py` → `figures/fig_head_gap.{png,pdf}`
