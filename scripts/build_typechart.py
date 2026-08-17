#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成属性克制关系 JSON（data/typechart.json）。

数据源：https://wiki.biligame.com/rocom/克制计算器 页面内嵌 typeEffectChart，
        双属性合成规则取自页面 JS（weak/vulnerable 合并计数：双命中 3.0/0.25，
        同时处于克制与被克制的系别互相抵消；进攻侧 strong/resist 用同一规则对称补齐）。

结构:
  elements   0-17 编号对应（普通0 草1 火2 水3 光4 地5 冰6 龙7 电8 毒9 虫10 武11 翼12 萌13 幽14 恶15 机械16 幻17）
  single     单属性每系：strong(克制)/resist(抵抗)/weak(被克制)/vulnerable(被抵抗) 均为 2.0/0.5 固定倍率
  dual       双属性每对（无序 a<b）：strong/resist/weak/vulnerable，weak/vulnerable 含倍率(2.0/3.0, 0.5/0.25)
"""
import json
import os
import re
from datetime import datetime, timezone
from collections import Counter

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(OUT)
DATA = os.path.join(ROOT, "data")

# 从页面内嵌 JS 提取的克制表（来源：克制计算器页面 typeEffectChart，2026-08-15 抓取）
TYPE_LIST = ["普通", "草", "火", "水", "光", "地", "冰", "龙", "电", "毒",
             "虫", "武", "翼", "萌", "幽", "恶", "机械", "幻"]
CHART_RAW = {
    "普通": {"strong": [], "resist": ["地", "幽", "机械"], "weak": ["武"], "vulnerable": ["幽"]},
    "草": {"strong": ["水", "光", "地"], "resist": ["火", "龙", "毒", "虫", "翼", "机械"],
           "weak": ["火", "冰", "毒", "虫", "翼"], "vulnerable": ["水", "地", "电", "光"]},
    "火": {"strong": ["草", "冰", "虫", "机械"], "resist": ["水", "地", "龙"],
           "weak": ["水", "地"], "vulnerable": ["草", "冰", "虫", "萌", "机械"]},
    "水": {"strong": ["火", "地", "机械"], "resist": ["草", "冰", "龙"],
           "weak": ["草", "电"], "vulnerable": ["火", "机械"]},
    "光": {"strong": ["幽", "恶"], "resist": ["草", "冰"],
           "weak": ["草", "幽"], "vulnerable": ["恶", "幻"]},
    "地": {"strong": ["火", "冰", "电", "毒"], "resist": ["草", "武"],
           "weak": ["草", "水", "冰", "武", "机械"], "vulnerable": ["普通", "火", "电", "毒", "翼"]},
    "冰": {"strong": ["草", "地", "龙", "翼"], "resist": ["火", "冰", "机械"],
           "weak": ["火", "地", "武", "机械"], "vulnerable": ["水", "冰", "光"]},
    "龙": {"strong": ["龙"], "resist": ["机械"],
           "weak": ["冰", "龙", "萌"], "vulnerable": ["草", "火", "水", "电", "翼"]},
    "电": {"strong": ["水", "翼"], "resist": ["草", "地", "龙", "电"],
           "weak": ["地"], "vulnerable": ["电", "翼", "机械"]},
    "毒": {"strong": ["草", "萌"], "resist": ["地", "毒", "幽", "机械"],
           "weak": ["地", "恶", "幻"], "vulnerable": ["草", "毒", "虫", "武", "萌"]},
    "虫": {"strong": ["草", "恶", "幻"], "resist": ["火", "毒", "武", "翼", "萌", "幽", "机械"],
           "weak": ["火", "翼"], "vulnerable": ["草", "武"]},
    "武": {"strong": ["普通", "地", "冰", "恶", "机械"], "resist": ["毒", "虫", "翼", "萌", "幽", "幻"],
           "weak": ["翼", "萌", "幻"], "vulnerable": ["地", "虫", "恶"]},
    "翼": {"strong": ["草", "虫", "武"], "resist": ["地", "龙", "电", "机械"],
           "weak": ["冰", "电"], "vulnerable": ["草", "虫", "武"]},
    "萌": {"strong": ["龙", "武", "恶"], "resist": ["火", "毒", "机械"],
           "weak": ["毒", "恶", "机械"], "vulnerable": ["虫", "武"]},
    "幽": {"strong": ["光", "幽", "幻"], "resist": ["普通", "恶"],
           "weak": ["光", "幽", "恶"], "vulnerable": ["普通", "毒", "虫", "武"]},
    "恶": {"strong": ["毒", "萌", "幽"], "resist": ["光", "武", "恶"],
           "weak": ["光", "虫", "武", "萌"], "vulnerable": ["幽", "恶"]},
    "机械": {"strong": ["地", "冰", "萌"], "resist": ["火", "水", "电", "机械"],
             "weak": ["火", "水", "武"],
             "vulnerable": ["普通", "草", "冰", "龙", "毒", "虫", "翼", "萌", "机械", "幻"]},
    "幻": {"strong": ["毒", "武"], "resist": ["光", "机械", "幻"],
           "weak": ["虫", "幽"], "vulnerable": ["武", "幻"]},
}

ELEMENT_ID = {name: i for i, name in enumerate(TYPE_LIST)}
ELEMENTS = {name: ELEMENT_ID[name] for name in TYPE_LIST}


def combine(a, b, field):
    """合并两个单系在 field 下的列表，返回 {type: count}（count 1 或 2）。"""
    return Counter((CHART_RAW[a].get(field) or []) + (CHART_RAW[b].get(field) or []))


def dual_view(a, b):
    """双属性（进攻/防守视角均按页面同一合成规则）。"""
    weak = combine(a, b, "weak")            # 受到伤害增加（被克制）
    vuln = combine(a, b, "vulnerable")      # 受到伤害降低（被抵抗）
    cancel = set(weak) & set(vuln)          # 同时克制与被克制 -> 抵消
    strong = combine(a, b, "strong")        # 造成伤害增加（克制）
    resist = combine(a, b, "resist")        # 造成伤害降低（抵抗）
    cancel2 = set(strong) & set(resist)     # 进攻侧对称抵消

    def items(cnt, single, double, skip):
        out = []
        for t, c in sorted(cnt.items()):
            if t in skip:
                continue
            out.append({"type": ELEMENT_ID[t], "value": double if c > 1 else single})
        return out

    return {
        "strong": items(strong, 2.0, 3.0, cancel2),
        "resist": items(resist, 0.5, 0.25, cancel2),
        "weak": items(weak, 2.0, 3.0, cancel),
        "vulnerable": items(vuln, 0.5, 0.25, cancel),
    }


def items_flat(cnt_list, value):
    return [{"type": ELEMENT_ID[x], "value": value} for x in sorted(cnt_list)]

single = {}
for t in TYPE_LIST:
    c = CHART_RAW[t]
    single[str(ELEMENT_ID[t])] = {
        "strong": items_flat(c.get("strong") or [], 2.0),
        "resist": items_flat(c.get("resist") or [], 0.5),
        "weak": items_flat(c.get("weak") or [], 2.0),
        "vulnerable": items_flat(c.get("vulnerable") or [], 0.5),
    }

dual = {}
for i in range(len(TYPE_LIST)):
    for j in range(i + 1, len(TYPE_LIST)):
        dual["%d-%d" % (i, j)] = dual_view(TYPE_LIST[i], TYPE_LIST[j])

doc = {
    "source": "https://wiki.biligame.com/rocom/克制计算器 (typeEffectChart)",
    "fetchedAt": datetime.now(timezone.utc).isoformat(),
    "elements": ELEMENTS,
    "single": single,
    "dual": dual,
}
os.makedirs(DATA, exist_ok=True)
with open(os.path.join(DATA, "typechart.json"), "w", encoding="utf-8") as f:
    json.dump(doc, f, ensure_ascii=False, indent=2)

print("elements:", len(ELEMENTS), "| single:", len(single), "| dual pairs:", len(dual))
print("wrote data/typechart.json")
