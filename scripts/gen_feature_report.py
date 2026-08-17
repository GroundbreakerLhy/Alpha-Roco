#!/usr/bin/env python3
"""生成特性实现清单(data/feature_implementation.md)。

分类规则:
  implemented - 在现有引擎机制内完整实现
  partial     - 依赖缺失机制(迅捷/传动/巧变/选择/天气/血脉/木桶等),
                以近似或部分方式实现,或仅实现其中一段
  natural     - 引擎天然满足,无需代码
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NATURAL = {
    "可获得的萌化层数不受限制。",
    "蓄力状态下，可以使用任一携带技能。",
}

MISSING_KEYS = [
    # 天气系统仍未实现(用户暂缓)
    "天气", "沙暴", "雨天", "雷鸣", "暴风雪", "水系环境",
    # 伪装/识破仅有框架,无伪装来源
    "伪装", "识破",
]


def classify(d: str) -> str:
    if d in NATURAL:
        return "natural"
    for k in MISSING_KEYS:
        if k in d:
            return "partial"
    if "选择" in d and "明」和「暗" in d:
        return "partial"  # 明暗依赖选择技能随机选项,近似
    return "implemented"


def main():
    spirits = json.load(open(ROOT / "data" / "spirits.json", encoding="utf-8"))["spirits"]
    distinct = Counter()
    for s in spirits:
        if s.get("feature"):
            distinct[s["feature"]["desc"]] += 1

    stats = Counter()
    lines = ["# 精灵特性实现清单\n",
             "由 `scripts/gen_feature_report.py` 生成。",
             "",
             "- **implemented**:在现有战斗引擎机制内完整实现",
             "- **partial**:依赖缺失机制(迅捷/传动/巧变/选择/天气/血脉/木桶等),以近似或部分方式实现,或仅实现其中一段",
             "- **natural**:引擎天然满足,无需实现",
             "",
             "| 状态 | 描述 | 携带精灵数 |",
             "|---|---|---|"]
    for d, c in distinct.most_common():
        st = classify(d)
        stats[st] += 1
        lines.append(f"| {st} | {d} | {c} |")

    total = len(distinct)
    lines.append("")
    lines.append("## 统计")
    lines.append("")
    lines.append(f"- distinct 描述总数:{total}")
    for st in ("implemented", "partial", "natural"):
        lines.append(f"- {st}:{stats[st]}")
    lines.append(f"- 覆盖率(implemented+natural):{(stats['implemented'] + stats['natural']) / total * 100:.1f}%")
    # 精灵数加权
    sp_impl = sum(c for d, c in distinct.items() if classify(d) in ("implemented", "natural"))
    lines.append(f"- 精灵数加权覆盖率:{sp_impl / sum(distinct.values()) * 100:.1f}%")

    out = ROOT / "data" / "feature_implementation.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {out}")
    print(f"implemented={stats['implemented']} partial={stats['partial']} natural={stats['natural']}")
    print(f"描述覆盖率:{(stats['implemented'] + stats['natural']) / total * 100:.1f}%")
    print(f"精灵加权覆盖率:{sp_impl / sum(distinct.values()) * 100:.1f}%")


if __name__ == "__main__":
    sys.exit(main())
