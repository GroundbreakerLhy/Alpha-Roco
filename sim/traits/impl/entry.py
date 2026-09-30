"""入场/离场类特性（on_entry / on_leave）。

- 200149 专注力    入场首回合，获得物攻+100%（首次行动——任意技能或聚能——后结束）。
- 200244 图书守卫者  入场时若己方魔力为1，获得双攻+100%。
- 200126 洁癖     离场后，自己的增益和减益会被更换入场的精灵继承。
- 200161 做噩梦    敌方精灵离场后，更换入场的精灵失去3能量。
- 200162 噼啪！   入场后首次行动，所选技能使用次数+1（迸发：skill_use_count）。
- 200164 电流刺激  入场后获得迸发：本次攻击技能威力+40（attack_power_flat）。
- 200216 珊瑚骨    敌方精灵离场时，自己获得全技能能耗-3（ENERGY_COST buff 叠加）。
- 200102 地脉      初始能量为0，入场前己方精灵每放1次地系技能，回复3能量
                   （场下积累次数，入场时能量=次数×3，初始为0）。
- 200109 守护者    己方其他精灵每有1层萌化，自己入场时全技能能耗-1
                   （入场时统计己方其他精灵萌化总层数 N，加 ENERGY_COST -N）。
- 200110 水翼推进  己方精灵每使用1次水系技能，自己入场时获得全技能能耗-1
                   （全程累计次数 charges，入场时刷新能耗 buff 为当前累计数，不清零不叠加）。
- 200118 结晶水    初始能量为0，入场前己方精灵每放1次冰系技能，回复3能量
                   （场下积累次数，入场时能量=次数×3，初始为0）。
- 200132 虫群鼓舞  队伍中每有1只其他的虫系精灵，自己入场时获得攻防速+10%
                   （入场时统计己方其他存活虫系精灵数 N，攻/防/速各 +10%×N）。
- 200142 瞳中倒影  自己或其他精灵离场时，自己与更换入场的精灵交换血量百分比
                   （任意精灵离场都触发，on_leave）。
- 200147 散热      初始能量为0，入场前己方精灵每放1次火系技能，回复3能量
                   （场下积累次数，入场时能量=次数×3，初始为0）。
- 200153 蒸汽膨胀  己方精灵每使用1次火系技能，自己入场时获得全技能威力+10。
- 200177 慢热型    初始能量为0，入场前己方精灵每成功应对1次，回复5能量。
- 200139 渴求      入场时获得50%吸血（LIFESTEAL 5 层，每层10%）。
- 200220 稀兽花宝  根据自身血脉，入场时对己方/敌方施加不同效果。

语义说明：
- 专注力：buff 为 NORMAL 时长，持续到首次行动（技能或聚能）之后才结束；下场/重新入场重新触发。
- 洁癖：离场事件在引擎清除普通 buff 之前广播，因此能拿到完整的增益/减益列表，全部复制给入场精灵。
- 地脉：on_skill_end 只积累"己方"（ctx.side==actor.side）的地系技能且自己在场下；入场时兑现。
- 特性带来的所有 buff 均带 source_kind="trait" 标记（不算普通"增益"，见 buffs.Buff.is_gain）。
"""

from __future__ import annotations

import copy

from ... import buffs as B
from ... import burst
from ... import marks as M
from ... import traits as T
from ...enums import Element as E
from ..base import TraitHandler
from ..registry import register

EARTH = E.EARTH  # 5 地
WATER = E.WATER  # 3 水
ICE = E.ICE  # 6 冰
BUG = E.BUG  # 10 虫
FIRE = E.FIRE  # 2 火


# ---------------- 200149 专注力 ----------------
class Focus(TraitHandler):
    trait_id = 200149
    name = "专注力"
    desc = "入场首回合，获得物攻+100%。"
    implemented = True

    def on_entry(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        # 重新入场/复活时清除可能残留的旧 buff 引用
        old = ctx.state_of("buff")
        if old is not None and old in ctx.actor.buffs:
            ctx.actor.buffs.remove(old)
        buff = B.Buff(
            buff_type=B.BuffType.ATK,
            value=10,
            duration=B.DurationKind.NORMAL,
            source_kind="trait",
        )
        ctx.actor.buffs.append(buff)
        ctx.set_state("buff", buff)

    def on_skill_end(self, ctx):
        # 首次使用技能后结束
        self._end(ctx)

    def on_charge(self, ctx):
        # 聚能（蓄能技能）同样视为使用技能，也消耗
        self._end(ctx)

    def on_leave(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        ctx.set_state("buff", None)

    def _end(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        buff = ctx.state_of("buff")
        if buff is None:
            return
        ctx.set_state("buff", None)
        if buff in ctx.actor.buffs:
            ctx.actor.buffs.remove(buff)

    def display(self, state, pet):
        if pet.trait_state.get("buff") is None:
            return None
        return [{"name": "物攻", "layers": 1, "per": "100%", "gain": True}]


# ---------------- 200177 慢热型 ----------------
class SlowWarm(TraitHandler):
    trait_id = 200177
    name = "慢热型"
    desc = "初始能量为0，入场前己方精灵每成功应对1次，回复5能量。"
    implemented = True

    def on_battle_start(self, ctx):
        ctx.actor.energy = 0
        ctx.set_state("counter_charges", 0)

    def on_counter(self, ctx):
        # 只积累自己入场前、己方其他精灵成功应对的次数。
        if ctx.side != ctx.actor.side or ctx.active:
            return
        ctx.add_state("counter_charges", 1)

    def on_entry(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        charges = ctx.state_of("counter_charges", 0)
        ctx.actor.energy = min(10, charges * 5)
        ctx.set_state("counter_charges", 0)


# ---------------- 200244 图书守卫者 ----------------
class LibraryGuardian(TraitHandler):
    trait_id = 200244
    name = "图书守卫者"
    desc = "入场时，若自己魔力值为1，自己获得双攻+100%。"
    implemented = True

    def on_entry(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        if ctx.state.magic[ctx.actor.side] != 1:
            return
        B.add_buff(ctx.actor, B.BuffType.ATK, 10, source_kind="trait")
        B.add_buff(ctx.actor, B.BuffType.SPATK, 10, source_kind="trait")


# ---------------- 200126 洁癖 ----------------
class Cleanliness(TraitHandler):
    trait_id = 200126
    name = "洁癖"
    desc = "离场后，自己的增益和减益会被更换入场的精灵继承。"
    implemented = True

    def on_leave(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        incoming = ctx.extra.get("incoming")
        if incoming is None:
            return
        for buff in list(ctx.actor.buffs):
            incoming.buffs.append(copy.copy(buff))


# ---------------- 200161 做噩梦 ----------------
class Nightmare(TraitHandler):
    trait_id = 200161
    name = "做噩梦"
    desc = "敌方精灵离场后，更换入场的精灵失去3能量。"
    implemented = True

    def on_leave(self, ctx):
        # 敌方精灵离场（subject=离场精灵，incoming=入场精灵）；自己/己方离场不触发
        if ctx.subject is None or ctx.subject.side == ctx.actor.side:
            return
        incoming = ctx.extra.get("incoming")
        if incoming is None:
            return
        incoming.energy = max(0, incoming.energy - 3)


# ---------------- 200162 噼啪！ ----------------
class Crackle(TraitHandler):
    trait_id = 200162
    name = "噼啪！"
    desc = "入场后首次行动，所选技能使用次数+1。"
    implemented = True

    def on_entry(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        burst.add_burst(ctx.actor, "skill_use_count", 1)


# ---------------- 200164 电流刺激 ----------------
class CurrentStimulus(TraitHandler):
    trait_id = 200164
    name = "电流刺激"
    desc = "携带的攻击技能获得迸发：威力+40。"
    implemented = True

    def on_entry(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        burst.add_burst(ctx.actor, "attack_power_flat", 40)


# ---------------- 200216 珊瑚骨 ----------------
class CoralBone(TraitHandler):
    trait_id = 200216
    name = "珊瑚骨"
    desc = "敌方精灵离场时，自己获得全技能能耗-3。"
    implemented = True

    def on_leave(self, ctx):
        # 自己必须在场；敌方精灵离场（subject=离场精灵）；己方离场不触发
        if not ctx.is_active():
            return
        if ctx.subject is None or ctx.subject.side == ctx.actor.side:
            return
        B.add_buff(ctx.actor, B.BuffType.ENERGY_COST, -3, source_kind="trait")
        ctx.add_state("triggers", 1)

    def display(self, state, pet):
        n = pet.trait_state.get("triggers", 0)
        if n <= 0:
            return None
        return [{"name": "能耗", "layers": n, "per": "3", "gain": True}]


# ---------------- 200102 地脉 ----------------
class EarthVein(TraitHandler):
    trait_id = 200102
    name = "地脉"
    desc = "初始能量为0，入场前己方精灵每放1次地系技能，回复3能量。"
    implemented = True

    def on_skill_end(self, ctx):
        # 场下积累：己方（side==actor.side）精灵使用地系技能；自己入场后不再积累
        if ctx.skill is None or ctx.skill.element != EARTH:
            return
        if ctx.side != ctx.actor.side:
            return
        if ctx.is_active():
            return
        ctx.add_state("charges", 1)

    def on_entry(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        # 入场兑现：能量 = 积累次数 × 3（初始能量为0，入场时从0起只加积累）
        charges = ctx.state_of("charges", 0)
        ctx.actor.energy = charges * 3
        ctx.set_state("charges", 0)

    def display(self, state, pet):
        n = pet.trait_state.get("charges", 0)
        if n <= 0:
            return None
        return [{"name": "积累", "layers": n, "per": "3", "gain": True}]


# ---------------- 200109 守护者 ----------------
class Guardian(TraitHandler):
    trait_id = 200109
    name = "守护者"
    desc = "己方其他精灵每有1层萌化，自己入场时全技能能耗-1。"
    implemented = True

    def on_entry(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        n = 0
        for i, pet in enumerate(ctx.state.teams[ctx.actor.side]):
            if pet is ctx.actor:
                continue
            if pet.hp <= 0:
                continue  # 已死亡的精灵不计入
            n += B.get_buff_value(pet, B.BuffType.CUTE)
        if n <= 0:
            return
        B.add_buff(ctx.actor, B.BuffType.ENERGY_COST, -n, source_kind="trait")
        ctx.set_state("layers", n)

    def display(self, state, pet):
        n = pet.trait_state.get("layers", 0)
        if n <= 0:
            return None
        return [{"name": "能耗", "layers": n, "per": "1", "gain": True}]


# ---------------- 200110 水翼推进 ----------------
class HydroPropulsion(TraitHandler):
    trait_id = 200110
    name = "水翼推进"
    desc = "己方精灵每使用1次水系技能，自己入场时获得全技能能耗-1。"
    implemented = True

    def on_skill_end(self, ctx):
        # 全程累计：己方精灵使用水系技能（自己在场与否都累计）
        if ctx.skill is None or ctx.skill.element != WATER:
            return
        if ctx.side != ctx.actor.side:
            return
        ctx.add_state("charges", 1)

    def on_entry(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        charges = ctx.state_of("charges", 0)
        if charges <= 0:
            return
        # 刷新为当前累计数：先移除旧的特性能耗 buff，再加新的（不叠加）
        ctx.actor.buffs = [
            b
            for b in ctx.actor.buffs
            if not (b.buff_type == B.BuffType.ENERGY_COST and b.source_kind == "trait")
        ]
        B.add_buff(ctx.actor, B.BuffType.ENERGY_COST, -charges, source_kind="trait")

    def display(self, state, pet):
        n = pet.trait_state.get("charges", 0)
        if n <= 0:
            return None
        return [{"name": "积累", "layers": n, "per": "1", "gain": True}]


# ---------------- 200118 结晶水 ----------------
class CrystalWater(TraitHandler):
    trait_id = 200118
    name = "结晶水"
    desc = "初始能量为0，入场前己方精灵每放1次冰系技能，回复3能量。"
    implemented = True

    def on_skill_end(self, ctx):
        # 场下积累：己方精灵使用冰系技能；自己入场后不再积累
        if ctx.skill is None or ctx.skill.element != ICE:
            return
        if ctx.side != ctx.actor.side:
            return
        if ctx.is_active():
            return
        ctx.add_state("charges", 1)

    def on_entry(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        charges = ctx.state_of("charges", 0)
        ctx.actor.energy = charges * 3
        ctx.set_state("charges", 0)

    def display(self, state, pet):
        n = pet.trait_state.get("charges", 0)
        if n <= 0:
            return None
        return [{"name": "积累", "layers": n, "per": "3", "gain": True}]


# ---------------- 200132 虫群鼓舞 ----------------
class SwarmCheer(TraitHandler):
    trait_id = 200132
    name = "虫群鼓舞"
    desc = "队伍中每有1只其他的虫系精灵，自己入场时获得攻防速+10%。"
    implemented = True

    def on_entry(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        n = 0
        for i, pet in enumerate(ctx.state.teams[ctx.actor.side]):
            if pet is ctx.actor:
                continue
            if BUG in pet.attributes:
                n += 1
        if n <= 0:
            return
        B.add_buff(ctx.actor, B.BuffType.ATK, n, source_kind="trait")
        B.add_buff(ctx.actor, B.BuffType.DEF, n, source_kind="trait")
        B.add_buff(ctx.actor, B.BuffType.SPEED_PERCENT, n, source_kind="trait")
        ctx.set_state("count", n)

    def display(self, state, pet):
        n = pet.trait_state.get("count", 0)
        if n <= 0:
            return None
        return [{"name": "攻防速", "layers": n, "per": "10%", "gain": True}]


# ---------------- 200142 瞳中倒影 ----------------
class MirrorReflection(TraitHandler):
    trait_id = 200142
    name = "瞳中倒影"
    desc = "自己或其他精灵离场时，自己与更换入场的精灵交换血量百分比。"
    implemented = True

    def on_leave(self, ctx):
        # 任意精灵（含敌方）离场都触发；与 incoming 交换血量百分比
        incoming = ctx.extra.get("incoming")
        if incoming is None:
            return
        own_pct = ctx.actor.hp / ctx.actor.max_hp
        inc_pct = incoming.hp / incoming.max_hp
        ctx.actor.hp = round(ctx.actor.max_hp * inc_pct)
        incoming.hp = round(incoming.max_hp * own_pct)


# ---------------- 200253 吉利丁片 ----------------
class GelatinSheet(TraitHandler):
    trait_id = 200253
    name = "吉利丁片"
    desc = "离场后，更换入场的精灵获得双防+20%且免疫冻结。"
    implemented = True

    def on_leave(self, ctx):
        # 自己离场时，更换入场的精灵获得双防+20%（各2层）与免疫冻结。
        # 效果作用于友方入场精灵（不是特性拥有者自身），用普通 buff 以便在
        # 该精灵的 buff 区正常显示（source_kind="trait" 会被客户端过滤）。
        if ctx.subject is not ctx.actor:
            return
        incoming = ctx.extra.get("incoming")
        if incoming is None:
            return
        B.add_buff(incoming, B.BuffType.DEF, 2)
        B.add_buff(incoming, B.BuffType.SPDEF, 2)
        B.add_buff(incoming, B.BuffType.FREEZE_IMMUNE, 1)


# ---------------- 200254 美拉德反应 ----------------
class MaillardReaction(TraitHandler):
    trait_id = 200254
    name = "美拉德反应"
    desc = "离场后，更换入场的精灵获得双攻+20%且免疫灼烧。"
    implemented = True

    def on_leave(self, ctx):
        # 自己离场时，更换入场的精灵获得双攻+20%（各2层）与免疫灼烧。
        # 效果作用于友方入场精灵，用普通 buff 以便正常显示（见 200253 说明）。
        if ctx.subject is not ctx.actor:
            return
        incoming = ctx.extra.get("incoming")
        if incoming is None:
            return
        B.add_buff(incoming, B.BuffType.ATK, 2)
        B.add_buff(incoming, B.BuffType.SPATK, 2)
        B.add_buff(incoming, B.BuffType.BURN_IMMUNE, 1)


# ---------------- 200202 茶多酚 ----------------
class TeaPolyphenol(TraitHandler):
    trait_id = 200202
    name = "茶多酚"
    desc = "离场后，更换入场的精灵回复20%生命且免疫寄生。"
    implemented = True

    def on_leave(self, ctx):
        # 自己离场时，更换入场的精灵回复20%最大生命并获得免疫寄生
        if ctx.subject is not ctx.actor:
            return
        incoming = ctx.extra.get("incoming")
        if incoming is None:
            return
        heal = int(incoming.max_hp * 0.2)
        incoming.hp = min(incoming.max_hp, incoming.hp + heal)
        B.add_buff(incoming, B.BuffType.LEECH_IMMUNE, 1)


# ---------------- 200163 快充 ----------------
class QuickCharge(TraitHandler):
    trait_id = 200163
    name = "快充"
    desc = "离场时回复10能量。"
    implemented = True

    def on_leave(self, ctx):
        # 自己离场时回复 10 能量（受能量上限修正，用 grant_energy 统一）
        if ctx.subject is not ctx.actor:
            return
        T.grant_energy(ctx.state, ctx.actor, 10)


# ---------------- 200153 蒸汽膨胀 ----------------
class SteamExpansion(TraitHandler):
    trait_id = 200153
    name = "蒸汽膨胀"
    desc = "己方精灵每使用1次火系技能，自己入场时获得全技能威力+10。"
    implemented = True

    def on_skill_end(self, ctx):
        if ctx.skill is None or ctx.skill.element != FIRE:
            return
        if ctx.side != ctx.actor.side:
            return
        ctx.add_state("charges", 1)

    def on_entry(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        old = ctx.state_of("buff")
        if old is not None and old in ctx.actor.buffs:
            ctx.actor.buffs.remove(old)
        charges = ctx.state_of("charges", 0)
        if charges <= 0:
            ctx.set_state("buff", None)
            return
        buff = B.Buff(
            buff_type=B.BuffType.SKILL_POWER_FLAT,
            value=charges * 10,
            duration=B.DurationKind.NORMAL,
            source_kind="trait",
        )
        ctx.actor.buffs.append(buff)
        ctx.set_state("buff", buff)

    def display(self, state, pet):
        n = pet.trait_state.get("charges", 0)
        if n <= 0:
            return None
        return [{"name": "技能威力", "layers": n, "per": "10", "gain": True}]


# ---------------- 200147 散热 ----------------
class HeatDissipation(TraitHandler):
    trait_id = 200147
    name = "散热"
    desc = "初始能量为0，入场前己方精灵每放1次火系技能，回复3能量。"
    implemented = True

    def on_skill_end(self, ctx):
        # 场下积累：己方精灵使用火系技能；自己入场后不再积累
        if ctx.skill is None or ctx.skill.element != FIRE:
            return
        if ctx.side != ctx.actor.side:
            return
        if ctx.is_active():
            return
        ctx.add_state("charges", 1)

    def on_entry(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        charges = ctx.state_of("charges", 0)
        ctx.actor.energy = charges * 3
        ctx.set_state("charges", 0)

    def display(self, state, pet):
        n = pet.trait_state.get("charges", 0)
        if n <= 0:
            return None
        return [{"name": "积累", "layers": n, "per": "3", "gain": True}]


# ---------------- 200139 渴求 ----------------
class Thirst(TraitHandler):
    trait_id = 200139
    name = "渴求"
    desc = "入场时获得50%吸血。"
    implemented = True

    def on_entry(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        B.add_buff(ctx.actor, B.BuffType.LIFESTEAL, 5, source_kind="trait")
        ctx.set_state("active", True)

    def display(self, state, pet):
        if not pet.trait_state.get("active"):
            return None
        return [{"name": "吸血", "layers": 5, "per": "10%", "gain": True}]


# ---------------- 200220 稀兽花宝 ----------------
class RareBeastFlower(TraitHandler):
    # 兽花蕾的特性给的 buff 不用标记 source_kind = "trait"
    trait_id = 200220
    name = "稀兽花宝"
    desc = "根据自己的血脉，入场时获得不同效果。"
    implemented = True

    def _self_buff(self, ctx, buff_type, value):
        B.add_buff(ctx.actor, buff_type, value)

    def _enemy_buff(self, ctx, target, buff_type, value):
        B.add_buff(
            target,
            buff_type,
            value,
            source_side=ctx.actor.side,
            source_pet=ctx.actor.name,
            source_kind="debuff",
        )

    def on_entry(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        target = ctx.opponent()
        if target is None:
            return

        bloodline = ctx.actor.bloodline
        if bloodline is None:
            bloodline = ctx.actor.attributes[0] if ctx.actor.attributes else None
        if bloodline is None:
            return

        display = []
        if bloodline == E.NORMAL:
            self._self_buff(ctx, B.BuffType.SKILL_POWER_FLAT, 40)
            display.append({"name": "技能威力", "layers": 1, "per": "40", "gain": True})
        elif bloodline == E.EARTH:
            self._enemy_buff(ctx, target, B.BuffType.SPEED, -6)
            self._enemy_buff(ctx, target, B.BuffType.HIT_COUNT_FLAT, -3)
            display.append(
                {"name": "敌方速度", "layers": 6, "per": "10", "gain": False}
            )
            display.append(
                {"name": "敌方连击数", "layers": 3, "per": "1", "gain": False}
            )
        elif bloodline == E.ICE:
            B.add_buff(
                target,
                B.BuffType.FREEZE,
                2,
                source_side=ctx.actor.side,
                source_pet=ctx.actor.name,
                source_kind="debuff",
            )
            display.append(
                {"name": "敌方冻结", "layers": 2, "per": "5%", "gain": False}
            )
        elif bloodline == E.CUTE:
            self._enemy_buff(ctx, target, B.BuffType.ATK, -6)
            self._enemy_buff(ctx, target, B.BuffType.SPATK, -6)
            display.append(
                {"name": "敌方双攻", "layers": 6, "per": "10%", "gain": False}
            )
        elif bloodline == E.FIRE:
            self._enemy_buff(ctx, target, B.BuffType.BURN, 6)
            display.append(
                {"name": "敌方灼烧", "layers": 6, "per": "2%", "gain": False}
            )
        elif bloodline == E.DRAGON:
            self._enemy_buff(ctx, target, B.BuffType.SPDEF, -8)
            display.append(
                {"name": "敌方魔防", "layers": 8, "per": "10%", "gain": False}
            )
        elif bloodline == E.MECH:
            self._self_buff(ctx, B.BuffType.DEF, 6)
            self._self_buff(ctx, B.BuffType.SPDEF, 6)
            display.append({"name": "双防", "layers": 6, "per": "10%", "gain": True})
        elif bloodline == E.WATER:
            self._self_buff(ctx, B.BuffType.ENERGY_COST, -2)
            display.append({"name": "能耗", "layers": 2, "per": "1", "gain": True})
        elif bloodline == E.ELECTRIC:
            self._self_buff(ctx, B.BuffType.SPEED, 10)
            display.append({"name": "速度", "layers": 10, "per": "10", "gain": True})
        elif bloodline == E.FANTASY:
            M.add_mark(ctx.state, target.side, 7, 2)
            display.append(
                {"name": "敌方星陨印记", "layers": 2, "per": "1", "gain": False}
            )
        elif bloodline == E.GRASS:
            heal = int(ctx.actor.max_hp * 0.2)
            ctx.actor.hp = min(ctx.actor.max_hp, ctx.actor.hp + heal)
            display.append({"name": "回复", "layers": 20, "per": "%", "gain": True})
        elif bloodline == E.POISON:
            self._enemy_buff(ctx, target, B.BuffType.POISON, 2)
            display.append(
                {"name": "敌方中毒", "layers": 2, "per": "3%", "gain": False}
            )
        elif bloodline == E.LIGHT:
            self._self_buff(ctx, B.BuffType.SPATK, 8)
            display.append({"name": "魔攻", "layers": 8, "per": "10%", "gain": True})
        elif bloodline == E.BUG:
            self._enemy_buff(ctx, target, B.BuffType.DEF, -8)
            display.append(
                {"name": "敌方物防", "layers": 8, "per": "10%", "gain": False}
            )
        elif bloodline == E.DARK:
            self._self_buff(ctx, B.BuffType.LIFESTEAL, 5)
            display.append({"name": "吸血", "layers": 5, "per": "10%", "gain": True})
        elif bloodline == E.FIGHT:
            self._self_buff(ctx, B.BuffType.ATK, 8)
            display.append({"name": "物攻", "layers": 8, "per": "10%", "gain": True})
        elif bloodline == E.GHOST:
            target.energy = max(0, target.energy - 2)
            display.append({"name": "敌方能量", "layers": 2, "per": "1", "gain": False})
        elif bloodline == E.WING:
            self._self_buff(ctx, B.BuffType.HIT_COUNT_FLAT, 3)
            display.append({"name": "连击数", "layers": 3, "per": "1", "gain": True})

        ctx.set_state("bloodline", int(bloodline))
        ctx.set_state("display", display)

    def display(self, state, pet):
        idx = state.active[pet.side]
        if idx < 0 or state.teams[pet.side][idx] is not pet:
            return None
        return pet.trait_state.get("display")


def register_entry() -> None:
    for cls in (
        Focus,
        SlowWarm,
        LibraryGuardian,
        Cleanliness,
        Nightmare,
        Crackle,
        CurrentStimulus,
        CoralBone,
        EarthVein,
        Guardian,
        HydroPropulsion,
        SteamExpansion,
        CrystalWater,
        SwarmCheer,
        MirrorReflection,
        HeatDissipation,
        Thirst,
        RareBeastFlower,
        GelatinSheet,
        MaillardReaction,
        TeaPolyphenol,
        QuickCharge,
    ):
        register(cls())


register_entry()
