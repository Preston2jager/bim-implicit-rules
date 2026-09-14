# 实验二：捷径审计与可推导性上限

**设定**：六窗规则数据（门位置 6 槽位）。对比两个数据版本下的 holdout 按规则分解：

- **泄漏版**（run `2026JUL110024`，41 copies）：`Name` 分类特征携带 Revit 实例号
  与 `_L####` 长度后缀。
- **去身份化版**（run `2026JUL112325_good_1`，81 copies）：非 IfcSpace 节点的
  Name 剥去 `:数字(_L数字)?` 尾缀。

**结论**：
1. 泄漏版的六窗规则 100%、单侧墙规则 100% 是**纯词表查表**（变换不变的捷径），
   不证明任何几何能力；唯一必须读几何的语义规则（马桶侧）在未见变换上 49.7% = 纯随机。
2. 去身份化后六窗/单侧墙掉到 70.1%/76.6%（真实几何能力），语义规则升到 64.5%
   （copies 翻倍首次超过随机水平）。
3. **可推导性上限 = 100%**：线性探针 z→质心只在训练 copies 上拟合，holdout 误差
   11.3 cm / R²=0.9994；"探针质心 + 拓扑 + 合法类别特征"的手工管线在全部 4824 张
   holdout 图上 100% 复现三种规则。因此缺口完全是端到端学习失败，而非数据缺陷。
   （验证脚本原件：scratchpad `verify_direction_derivable.py`；数值存于
   `data/per_rule.json` 的 `ceiling` 字段。）

**文件**：
- `extract_data.py` → `data/per_rule.json`（按 L/R>1m 从文件名归类三种规则）
- `make_figures.py` → `figures/fig_rule_breakdown.{png,pdf}`
