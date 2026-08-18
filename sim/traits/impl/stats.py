"""第 1 批-b：属性/速度修正（modify_stat / modify_speed）。

- 200090 壮胆       队伍存在虫系精灵，自己获得双攻+50%。
- 200140 悲悯       己方队伍中每有1只力竭的精灵，自己获得双攻+30%。（原第 0 批测试队伍，归入此类）
- 200091 囤积       每有1能量，获得双防+10%。
- 200094 保守派     总技能能耗小于4时，自己获得双防+80%。
- 200309 守护之心   双方场上每有1种不同的增益，自己获得物防+20%。
- 200310 和弦共振   双方场上每有1种不同的印记，自己获得魔攻+50%。
- 280024 悼亡       双方队伍中每有1只力竭的精灵，自己获得双攻+30%。
- 200298 流沙统治者 天气为沙暴时，自己获得速度+50。

语义说明：
- 守护之心/变形活画数的是"增益"，用 Buff.is_gain()（特性来源不算增益）。
- 和弦共振数印记种类（双方正/负槽非空印记 id 去重）。
"""

from __future__ import annotations

from ...enums import Element as E
from ..registry import register
from ..base import TraitHandler

BUG = E.BUG        # 10 虫
SANDSTORM = 2      # 天气：沙暴


def _self(ctx):
    return ctx.target is ctx.actor


def _active_pets(ctx):
    pets = []
    for side in ("A", "B"):
        idx = ctx.state.active[side]
        if idx >= 0:
            pets.append(ctx.state.teams[side][idx])
    return pets


# ---------------- 200140 悲悯 ----------------
class Compassion(TraitHandler):
    trait_id = 200140
    name = "悲悯"
    desc = "己方队伍中每有1只力竭的精灵，自己获得双攻+30%。"
    implemented = True

    def modify_stat(self, ctx, stat):
        if not _self(ctx) or stat not in ("atk", "spatk"):
            return 0.0
        n = sum(1 for p in ctx.state.teams[ctx.actor.side] if p.hp <= 0)
        return 0.3 * n


# ---------------- 200090 壮胆 ----------------
class BraveHeart(TraitHandler):
    trait_id = 200090
    name = "壮胆"
    desc = "队伍存在虫系精灵，自己获得双攻+50%。"
    implemented = True

    def modify_stat(self, ctx, stat):
        if not _self(ctx) or stat not in ("atk", "spatk"):
            return 0.0
        has_bug = any(BUG in p.attributes for p in ctx.state.teams[ctx.actor.side])
        return 0.5 if has_bug else 0.0


# ---------------- 200091 囤积 ----------------
class Hoard(TraitHandler):
    trait_id = 200091
    name = "囤积"
    desc = "每有1能量，获得双防+10%。"
    implemented = True

    def modify_stat(self, ctx, stat):
        if not _self(ctx) or stat not in ("def", "spdef"):
            return 0.0
        return 0.1 * ctx.actor.energy


# ---------------- 200094 保守派 ----------------
class Conservative(TraitHandler):
    trait_id = 200094
    name = "保守派"
    desc = "总技能能耗小于4时，自己获得双防+80%。"
    implemented = True

    def modify_stat(self, ctx, stat):
        if not _self(ctx) or stat not in ("def", "spdef"):
            return 0.0
        total = sum(s.energy_cost for s in ctx.actor.skills)
        return 0.8 if total < 4 else 0.0


# ---------------- 200309 守护之心 ----------------
class GuardianHeart(TraitHandler):
    trait_id = 200309
    name = "守护之心"
    desc = "双方场上每有1种不同的增益，自己获得物防+20%。"
    implemented = True

    def modify_stat(self, ctx, stat):
        if not _self(ctx) or stat != "def":
            return 0.0
        types = set()
        for pet in _active_pets(ctx):
            types.update(b.buff_type for b in pet.buffs if b.is_gain())
        return 0.2 * len(types)


# ---------------- 200310 和弦共振 ----------------
class ChordResonance(TraitHandler):
    trait_id = 200310
    name = "和弦共振"
    desc = "双方场上每有1种不同的印记，自己获得魔攻+50%。"
    implemented = True

    def modify_stat(self, ctx, stat):
        if not _self(ctx) or stat != "spatk":
            return 0.0
        ids = set()
        for side in ("A", "B"):
            for slot in ("positive", "negative"):
                mark = ctx.state.marks[side][slot]
                if mark is not None:
                    ids.add(mark["id"])
        return 0.5 * len(ids)


# ---------------- 280024 悼亡 ----------------
class Mourning(TraitHandler):
    trait_id = 280024
    name = "悼亡"
    desc = "双方队伍中每有1只力竭的精灵，自己获得双攻+30%。"
    implemented = True

    def modify_stat(self, ctx, stat):
        if not _self(ctx) or stat not in ("atk", "spatk"):
            return 0.0
        n = 0
        for side in ("A", "B"):
            n += sum(1 for p in ctx.state.teams[side] if p.hp <= 0)
        return 0.3 * n


# ---------------- 200298 流沙统治者 ----------------
class SandRuler(TraitHandler):
    trait_id = 200298
    name = "流沙统治者"
    desc = "天气为沙暴时，自己获得速度+50。"
    implemented = True

    def modify_speed(self, ctx):
        if not _self(ctx):
            return 0
        return 50 if ctx.weather == SANDSTORM else 0


def register_batch1_stats() -> None:
    for cls in (Compassion, BraveHeart, Hoard, Conservative, GuardianHeart,
                ChordResonance, Mourning, SandRuler):
        register(cls())


register_batch1_stats()
