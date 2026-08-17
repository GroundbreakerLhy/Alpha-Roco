#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 30 种性格数据（data/natures.json）。

来源：biligame wiki 性格页 + RocomUID nature_map 交叉核对（两源完全一致）。
每性格：up=加成属性 +10%，down=减损属性 -10%。
"""
import json
import os
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# (名称, 加成, 减损)
NATURES = [
    ("大胆", "物攻", "物防"), ("固执", "物攻", "魔攻"), ("调皮", "物攻", "魔防"),
    ("勇敢", "物攻", "速度"), ("逞强", "物攻", "生命"),
    ("稳重", "物防", "物攻"), ("天真", "物防", "魔攻"), ("懒散", "物防", "魔防"),
    ("悠闲", "物防", "速度"), ("坦率", "物防", "生命"),
    ("聪明", "魔攻", "物攻"), ("专注", "魔攻", "物防"), ("偏执", "魔攻", "魔防"),
    ("冷静", "魔攻", "速度"), ("理性", "魔攻", "生命"),
    ("警惕", "魔防", "物攻"), ("温顺", "魔防", "物防"), ("害羞", "魔防", "魔攻"),
    ("慎重", "魔防", "速度"), ("焦虑", "魔防", "生命"),
    ("胆小", "速度", "物攻"), ("急躁", "速度", "物防"), ("开朗", "速度", "魔攻"),
    ("莽撞", "速度", "魔防"), ("热情", "速度", "生命"),
    ("沉默", "生命", "物攻"), ("忧郁", "生命", "物防"), ("平和", "生命", "魔攻"),
    ("粗心", "生命", "魔防"), ("踏实", "生命", "速度"),
]

# 对应 spirits.json stats 的键
STAT_KEY = {"生命": "hp", "物攻": "atk", "魔攻": "spatk",
            "物防": "def", "魔防": "spdef", "速度": "speed"}

doc = {
    "source": "biligame wiki 性格页 + RocomUID nature_map",
    "fetchedAt": datetime.now(timezone.utc).isoformat(),
    "count": len(NATURES),
    "natures": [{"id": i, "name": n, "up": STAT_KEY[u], "down": STAT_KEY[d]}
                for i, (n, u, d) in enumerate(NATURES)],
}
os.makedirs(DATA, exist_ok=True)
with open(os.path.join(DATA, "natures.json"), "w", encoding="utf-8") as f:
    json.dump(doc, f, ensure_ascii=False, indent=2)
print("natures:", len(NATURES))
print("wrote data/natures.json")
