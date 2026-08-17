#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""根据最终 JSON 结构反向整理的生成脚本（数据源：biligame wiki PetData 模块，S3）。

输入（.cache/，原始 Lua 模块）:
  PetDataCore.txt            精灵基础：编号/游戏id/名称/形态/属性/种族值/特性引用
  PetData_SkillCatalog.txt   全部技能与特性定义
  PetData_Learnsets.txt      精灵 -> 学习表
  PetData_LearnsetCatalog.txt 学习表：ns/bs/ss/fs
  PetData_Evolution.txt      进化链（含首领化分支）

输出（data/）:
  spirits.json  596 只精灵，按 no 排序
  skills.json   557 个技能

属性统一为 0 起始编号：
  普通0 草1 火2 水3 光4 地5 冰6 龙7 电8 毒9 虫10 武11 翼12 萌13 幽14 恶15 机械16 幻17
"""
import json
import os
import re
import sys
from datetime import datetime, timezone

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(SCRIPTS)                      # 项目根目录
CACHE = os.path.join(OUT, ".cache")                 # 原始模块
DATA = os.path.join(OUT, "data")                    # 输出目录
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
from parse_lua import load_lua  # noqa: E402

core = load_lua(os.path.join(CACHE, "PetDataCore.txt"))
skill_cat = load_lua(os.path.join(CACHE, "PetData_SkillCatalog.txt"))
pet2learnset = load_lua(os.path.join(CACHE, "PetData_Learnsets.txt"))
learnset_cat = load_lua(os.path.join(CACHE, "PetData_LearnsetCatalog.txt"))
evo_mod = load_lua(os.path.join(CACHE, "PetData_Evolution.txt"))
skill_cat.pop("_meta", None)
evo_mod.pop("_meta", None)

ELEMENT_ID = {"普通": 0, "草": 1, "火": 2, "水": 3, "光": 4, "地": 5, "冰": 6,
              "龙": 7, "电": 8, "毒": 9, "虫": 10, "武": 11, "翼": 12, "萌": 13,
              "幽": 14, "恶": 15, "机械": 16, "幻": 17}
CATEGORY_ID = {"物攻": 0, "魔攻": 1, "防御": 2, "状态": 3}


# ---------------------------------------------------------------- helpers --
def element_id(e):
    e = re.sub(r"(系|别)$", "", e or "")
    return ELEMENT_ID.get(e, e)  # 未匹配则原样保留


def no_of(pet):
    hi = (pet.get("hb") or {}).get("i")
    if hi and hi.startswith("handbook_"):
        return str(int(hi.replace("handbook_", ""))).zfill(3)
    return None


def game_id(pet):
    m = re.search(r"(\d+)$", (pet.get("img") or {}).get("hd") or "")
    return int(m.group(1)) if m else None


def form_type_of(pet):
    f = pet.get("f") or ""
    if f == "首领形态" or pet.get("le"):
        return "首领形态"
    if f:
        return "特殊形态"
    return "主形态"


# ----------------------------------------------------------- skill catalog --
def norm_skill(rec):
    cat = rec.get("category")
    if cat == "攻击":
        cat = rec.get("damage_class") or cat  # 攻击 -> 物攻/魔攻
    return {
        "id": rec.get("id"),
        "name": rec.get("name"),
        "element": element_id(rec.get("element")),
        "category": CATEGORY_ID.get(cat, cat),
        "energyCost": rec.get("energy"),
        "power": rec.get("power"),
        "desc": rec.get("desc"),
    }


VOID_SKILLS = {7800551, 7800552, 7800553, 7800554}  # "空"系隐藏占位技能，不存在，删除

skills_by_ref = {}      # skill_XXXXXX -> 技能定义
features_by_ref = {}    # skill_XXXXXX -> 特性定义
skills_by_id = {}       # 真实技能 id -> 定义
for ref, rec in skill_cat.items():
    n = norm_skill(rec)
    if rec.get("id") in VOID_SKILLS:
        continue  # 删除不存在的"空"系技能
    if rec.get("category") == "特性":
        features_by_ref[ref] = n
    else:
        skills_by_ref[ref] = n
        skills_by_id.setdefault(n["id"], n)

# 通用自带技能：蓄能（状态，恢复5能量），所有精灵默认携带
BUILTIN_SKILL = {"id": 9999999, "name": "蓄能", "element": 0, "category": 3,
                 "energyCost": 0, "power": 0, "desc": "恢复5能量。"}
skills_by_id[BUILTIN_SKILL["id"]] = BUILTIN_SKILL


# ------------------------------------------------------------- evolution --
WIKI_ID_MAP = {pid: {"id": game_id(p), "no": no_of(p)} for pid, p in core.items()
               if isinstance(p, dict) and pid != "_meta"}


def evo_for_pet(pet):
    out = []
    for evg in pet.get("evg") or []:
        e = evo_mod.get(evg)
        if not e:
            continue
        chain = []
        for step in e.get("chain") or []:
            m = WIKI_ID_MAP.get(step.get("id"), {})
            chain.append({
                "id": m.get("id"), "no": m.get("no"),
                "name": step.get("name"),
                "stage": step.get("stage"), "level": step.get("level") or None,
                "types": [element_id(x) for x in (step.get("types") or [])],
            })
        lords = []
        for lb in e.get("lord_branches") or []:
            m = WIKI_ID_MAP.get(lb.get("id"), {})
            lords.append({
                "id": m.get("id"), "no": m.get("no"),
                "name": lb.get("name"),
                "form": lb.get("form"), "cond": lb.get("cond"),
                "types": [element_id(x) for x in (lb.get("types") or [])],
            })
        out.append({"name": e.get("name"), "chain": chain, "lordBranches": lords})
    return out


# ---------------------------------------------------------------- spirits --
spirits = []
for pid, pet in core.items():
    if pid == "_meta" or not isinstance(pet, dict):
        continue

    st = pet.get("st") or {}
    stats = {"hp": st.get("hp"), "atk": st.get("at"), "spatk": st.get("sa"),
             "def": st.get("df"), "spdef": st.get("sd"), "speed": st.get("se")}

    ls = learnset_cat.get(pet2learnset.get(pid) or "") or {}
    normal = [skills_by_ref.get(i.get("sk"), {}).get("id") for i in ls.get("ns") or []]
    normal.insert(0, BUILTIN_SKILL["id"])  # 所有精灵自带蓄能
    bloodline = [{"skillId": skills_by_ref.get(i.get("sk"), {}).get("id"),
                  "bloodline": element_id(i.get("bl"))} for i in ls.get("bs") or []]
    skillstone = [skills_by_ref.get(s, {}).get("id") for s in ls.get("ss") or []]
    if form_type_of(pet) == "首领形态":
        bloodline = []  # 首领形态无血脉技能

    feat = features_by_ref.get(ls.get("fs") or pet.get("fs")) or {}
    if feat:
        feat = {"id": feat.get("id"), "desc": feat.get("desc")}

    tp = pet.get("tp") or []
    spirits.append({
        "no": no_of(pet),
        "id": game_id(pet),
        "name": pet.get("t") or pet.get("n"),
        "form": pet.get("f") or None,
        "formType": form_type_of(pet),
        "stage": pet.get("sg"),
        "mainElement": element_id(tp[0]) if tp else None,
        "subElement": element_id(tp[1]) if len(tp) > 1 else None,
        "stats": stats,
        "feature": feat or None,
        "skills": {"normal": normal, "bloodline": bloodline, "skillstone": skillstone},
        "evolution": evo_for_pet(pet),
    })

spirits.sort(key=lambda s: (s["no"] is None, int(s["no"]) if s["no"] else 0))
sk_list = sorted(skills_by_id.values(), key=lambda x: x["id"] or 0)

# ---------------------------------------------------------------- outputs --
os.makedirs(DATA, exist_ok=True)
now = datetime.now(timezone.utc).isoformat()
with open(os.path.join(DATA, "spirits.json"), "w", encoding="utf-8") as f:
    json.dump({"source": "biligame wiki Module:PetData/* (S3)",
               "fetchedAt": now, "count": len(spirits), "spirits": spirits},
              f, ensure_ascii=False, indent=2)
with open(os.path.join(DATA, "skills.json"), "w", encoding="utf-8") as f:
    json.dump({"source": "biligame wiki Module:PetData/SkillCatalog",
               "fetchedAt": now, "count": len(sk_list),
               "categories": CATEGORY_ID,   # 物攻0 魔攻1 防御2 状态3
               "skills": sk_list},
              f, ensure_ascii=False, indent=2)

print("spirits:", len(spirits), "| skills:", len(sk_list),
      "| features:", len(features_by_ref))
print("wrote data/spirits.json, data/skills.json")
