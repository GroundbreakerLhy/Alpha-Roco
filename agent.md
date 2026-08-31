# Roco 模拟器项目说明

## 1. 项目目标

这是一个无前端、服务端/客户端分离、面向 AI 模型交互的洛克王国世界对战模拟器。

当前主要运行形态：

- 无头（headless）回合制战斗
- 服务端 + 两个客户端
- 目标规则为 6v6 固定对战
- 客户端通过 JSON/TCP 与服务端交互
- 战斗规则集中在 `sim/` 中，由 `server.py` 驱动

> 注意：`roco-battle-simulator/` 是另一套较旧/较大型的模拟器实现。当前根目录的网络对战流程使用的是 `sim/`，不要将两套实现混为一谈。

---

## 2. 快速运行

在项目根目录 `/Users/groundbreaker/Roco` 下执行。

```bash
# 终端 1：启动服务端
python server.py

# 终端 2：手动客户端
python client.py --team data/teams/team0.json

# 终端 3：自动客户端
python client.py --auto --team data/teams/team1.json
```


服务端默认监听：

```text
127.0.0.1:5000
```

战斗日志写入：

```text
logs/battle/server.log
```

---

## 3. 目录结构

```text
server.py                 # TCP 对战服务端，驱动 sim.battle
client.py                 # 命令行客户端，负责显示状态和提交行动
agent.md                  # 项目运行逻辑与开发说明

data/
  spirits.json            # 精灵、属性、基础种族值、特性、进化链
  skills.json             # 技能名称、属性、类别、能耗、威力、描述
  typechart.json          # 属性克制关系
  natures.json            # 性格及属性倍率
  buff.json               # buff 展示和规则数据
  marks.json              # 印记数据
  weather.json            # 天气数据
  resonance.json          # 共鸣魔法数据
  traits.json             # 231 个特性及完成/测试标记
  teams/                  # 队伍配置
    xxx.json              # {"resonance": 0/1/2, team": [...] 可选}

sim/
  __init__.py             # 模拟器包入口
  battle.py               # 战斗状态机、行动结算、胜负、状态序列化
  models.py               # BattlePet / BattleSkill / BattleState / Action；BattleSkill.swift 为迅捷标记
  data_loader.py          # JSON 数据加载、队伍校验、面板计算、实体构造
  damage.py               # 攻击伤害计算
  buffs.py                # buff/debuff、状态效果、回合结束结算
  marks.py                # 正负印记槽及印记结算
  counter.py              # 应对类别、强制先手、应对统计
  skill_utils.py          # 技能类别、能耗、属性和应对辅助函数
  enums.py                # Element、SkillCategory、血脉等枚举常量
  weather.py              # 雨天、暴风雪、沙暴、雷鸣
  resonance.py            # 愿力冲击、进化之力、光合治愈
  burst.py                # 入场后首次行动消耗的迸发效果
  evolution.py            # 普通进化、首领化、萌化退化
  typechart.py            # 属性克制查询
  traits/
    README.md             # 特性系统设计和未建模机制说明
    base.py               # TraitContext / TraitHandler
    registry.py           # trait_id -> handler 注册表
    __init__.py           # 特性广播、数值查询、能量和形态绑定接口
    impl/                 # 已实现特性的 handler，未实现特性为 no-op
```

---

## 4. 系统边界和调用关系

```text
server.py
  ├─ 接受 A/B 两个客户端连接
  ├─ 接收双方队伍配置
  ├─ data_loader.make_battle_pet()
  ├─ battle.create_team_battle()
  ├─ battle.state_to_dict() → 发送初始状态
  └─ 循环接收双方 Action
       └─ battle.step(state, action_a, action_b)
            ├─ 回合开始
            ├─ 共鸣魔法
            ├─ 换人
            ├─ 先手/应对判定
            ├─ 执行动作
            ├─ 力竭判定
            ├─ 回合结束结算
            └─ 返回 BattleState
                 └─ state_to_dict() → 发回双方客户端
```

`sim/` 不负责用户输入，也没有独立的主程序入口。它是一个可被服务端或其他程序直接调用的战斗规则库。

---

## 5. 核心数据模型

定义在 `sim/models.py`。

### 5.1 `BattlePet`

一只战斗精灵的运行时状态，包括：

- 身份：`side`、`spirit_id`、`name`、`level`
- 面板：`stats`、`ivs`、`nature`
- 生命和能量：`hp`、`max_hp`、`energy`
- 技能：`skills`
- 技能运行标记：`BattleSkill.swift` 表示迅捷，通过 `skill_utils.is_swift(skill)` 查询；离场不是技能静态属性，而是 `_resolve_action()` 中本次行动的局部结算结果，不从 `desc` 文本推断
- 属性和血脉：`attributes`、`bloodline`
- buff/debuff：`buffs`
- 防御冷却：`defense_cooldowns`
- 迸发：`bursts`
- 超载：`overload_next`、`overload_current`
- 特性：`trait_id`、`trait_state`
- 应对统计：`counter_stats`

### 5.2 `BattleState`

整场战斗的全局状态，包括：

- 双方队伍：`teams["A"]`、`teams["B"]`
- 当前出战下标：`active`
- 双方魔力：`magic`
- 正面/负面印记：`marks`
- 天气：`weather`
- 共鸣魔法、使用次数和冷却
- 当前回合：`turn`
- 本回合日志：`log`
- 胜者：`winner`
- 是否已广播首发入场：`entry_done`
- 特性请求换人：`pending_switch`

### 5.3 `Action`

客户端提交的行动：

- `kind`: `skill`、`charge`、`switch`、`flee`
- `skill_index`: 技能下标
- `pet_index`: 换人目标下标
- `magic_id`: 共鸣魔法 ID
- `magic_branch`: 首领化分支

---

## 6. 战斗初始化

### 6.1 队伍构造

`server.py` 读取队伍 JSON 后，通过 `data_loader.make_battle_pet()` 构造运行时精灵：

1. 根据名字查找 `spirits.json`。
2. 根据技能名查找 `skills.json`。
3. 计算最终六维属性。
4. 设置初始生命为最大生命。
5. 设置初始能量为 `10`。
6. 设置属性、血脉和特性 ID。
7. 选择最多 4 个普通技能。

### 6.2 面板计算

普通属性大致为：

```text
(round_half_up(1.1 × (base + 3 × IV)) + 10) × 性格倍率 + 50
```

HP 大致为：

```text
(70 + (0.01 × base_hp + 0.005 × IV × 6) × 170) × 性格倍率 + 100
```

性格倍率：

- 提升属性：`1.2`
- 降低属性：`0.9`
- 其他属性：`1.0`

### 6.3 `create_team_battle()`

初始化：

- 双方当前出战精灵默认为第 0 只
- 双方魔力均为 `4`
- 回合初始值为 `1`
- 随机决定 `home_side`
- 初始化正负印记槽、天气、共鸣状态
- 对默认首发精灵执行一次技能传动排序

---

## 7. `battle.step()` 回合流程

`sim/battle.py` 中的 `step()` 是核心状态机。

### 7.1 回合开始

1. 如果已有胜者，直接返回。
2. `state.turn += 1`。
3. 清空上一回合日志。
4. 首次战斗时补发首发精灵的 `entry` 事件。
5. 广播 `turn_start` 特性事件，特性可以读取双方已选行动。
6. 清除本回合防御减伤。
7. 将 `overload_next` 转为本回合的 `overload_current`。

### 7.2 逃跑

如果任一方提交 `flee`：

- 该方立即失败
- 另一方获胜
- 本回合结束

### 7.3 共鸣魔法

共鸣魔法不占用普通回合行动，双方行动都收到后，在普通动作前结算。

| ID | 名称 | 行为 |
|---:|---|---|
| `0` | 愿力冲击 | 临时替换 1 号技能，最多使用 2 次 |
| `1` | 进化之力 | 首领化或指定首领分支 |
| `2` | 光合治愈 | 回复当前精灵 50% 最大生命 |

愿力冲击会在回合结束时恢复原技能。若对手本回合使用状态行动，愿力冲击威力提高。

### 7.4 换人优先

主动换人优先于攻击、聚能和状态动作。

主动换人流程：

1. 检查是否被禁足。
2. 广播 `leave` 事件。
3. 清除离场精灵的普通 buff。
4. 清除离场精灵的迸发。
5. 更新 `state.active`。
6. 对入场精灵调用 `switch_in(active_switch=True)`。
7. 处理入场印记和 `entry` 特性。
8. 按技能槽顺序查找第一个能量足够、且 `skill_utils.is_swift(skill)` 为真的技能并自动释放。

精灵力竭后不会在 `step()` 内等待选择，而是将 `active` 设为 `-1`，由 `server.py` 在回合外要求客户端选择替补。

### 7.5 先手判定

若双方都有普通行动，排序为：

1. 应对成功的一方强制先手。
2. 有效速度高者先手。
3. 速度相同则随机决定。

有效速度由以下内容共同决定：

```text
基础速度
+ 速度 buff
+ 印记修正
+ 特性速度修正
```

### 7.6 重复行动

`overload_current` 或入场迸发可以让一次技能/聚能被重复执行多次。

每次重复行动都会重新检查：

- 是否仍然存活
- 能量是否足够
- 是否因力竭或胜负而中断

---

## 8. 单个行动的处理

### 8.1 聚能

聚能会：

1. 标记精灵已经行动。
2. 消耗当前迸发。
3. 回复 `5` 点能量。
4. 通过特性查询能量上限。
5. 广播 `charge` 和 `energy_gain`。

默认能量上限为 `10`。特性“多人宿舍”可以突破上限。

### 8.2 技能可用性

技能使用前检查：

- 技能下标是否合法
- 特性是否允许使用该技能
- 实际能耗是否足够
- 防御技能是否处于冷却
- 是否有特性可以用生命替代不足的能量

实际能耗由以下因素叠加：

```text
技能原始能耗
+ buff/debuff 能耗修正
+ 天气修正
+ 印记修正
+ 特性修正
- 迸发临时减耗
```

能量不足时：

- “石头大餐”：消耗当前生命补充能量
- “盛宴”：消耗最大生命补充能量

### 8.3 技能开始事件

扣除实际能耗后、技能效果前广播 `skill_start`。

可在这里触发：

- 附加中毒、灼烧、冻结
- 生成星陨印记
- 改变技能属性
- 修改本次技能威力或其他数值

### 8.4 攻击技能

类别 `0` 为物理攻击，类别 `1` 为魔法攻击。

伤害计算主要由 `sim/damage.py` 完成：

```text
攻击属性
× 属性 buff
× 特性属性修正
× 技能最终威力
× 本系加成
× 属性克制倍率
× 等级系数
÷ 防御属性
× 防御减伤
× 造成伤害修正
× 受到伤害修正
```

主要规则：

- 同属性技能获得 `1.25` 本系加成。
- 雨天提升水系攻击技能威力。
- 印记、buff、迸发和特性共同影响技能威力。
- 连击次数由 buff、特性、迸发和敌方光环共同决定。
- 伤害计算结果取整。

伤害结算后可能触发：

```text
attack
  → take_damage
  → kill
```

攻击者之后按照实际造成伤害计算吸血。

### 8.5 致命伤害

如果预计伤害足以击杀目标，会先广播 `lethal` 事件。

- 免死特性可以阻止本次伤害。
- 没有免死效果时才真正扣除 HP。

### 8.6 防御技能

类别 `2` 的技能：

1. 从技能描述中解析 `减伤N%`。
2. 设置当前回合的 `defense_reduction`。
3. 记录本回合使用过防御。
4. 回合结束时让所有防御技能进入冷却。

任意防御技能使用后，该精灵的所有防御技能都会进入冷却。

### 8.7 状态技能

类别 `3` 的技能目前主要广播 `status_skill`，通用效果尚未完整实现。

---

## 9. 力竭与胜负

精灵 HP 小于等于 0 时，`_apply_faint()` 会：

1. 将当前出战位置设为 `-1`。
2. 对该方魔力减 `1`。
3. 广播 `faint` 事件。
4. 若魔力归零，判定对方获胜。

可能影响力竭结算的特性包括：

- 诈死：返还刚扣除的魔力
- 付给恶魔的赎价：额外扣除魔力
- 复活类特性：接口已预留，尚未全部实现

攻击后和回合结束持续伤害后都会检查力竭。

### 回合上限

`MAX_TURN = 50`。

当进入第 51 回合结算时：

1. 魔力值高者胜。
2. 魔力相同则比较双方队伍剩余生命总和。
3. 生命总和也相同则平局。

---

## 10. 回合结束结算

回合结束按主客场顺序执行：

```python
["A", "B"] if state.home_side == "A" else ["B", "A"]
```

结算顺序：

1. 防御冷却
2. 印记回合效果
3. buff/debuff 回合效果和临时 buff 过期
4. 共鸣冷却、愿力技能恢复
5. 天气效果和天气持续时间
6. 特性 `round_end`
7. 清理本回合 `overload_current`
8. 处理持续伤害导致的力竭
9. 处理特性请求换人
10. 回合上限判定
11. 下一回合开始前执行技能传动重排

### 状态类 buff

实现于 `sim/buffs.py`：

- 中毒：最大生命的 `3% × 层数`
- 灼烧：最大生命的 `2% × 层数`，结算后层数减半
- 寄生：最大生命的 `2% × 层数`，并给来源回复等量生命
- 冻结：按最大生命的 `5% × 层数` 检查是否力竭，不直接造成普通伤害
- 眩晕：下一回合无法行动，回合结束减少一层
- 禁足：无法主动换人，回合结束减少持续值
- 引电：达到 2 层时立即触发电系伤害并清除

---

## 11. Buff、印记、天气和迸发

### 11.1 Buff

`sim/buffs.py` 使用统一的 `Buff` 对象。

持续时间：

- `permanent`：永久存在，换人不清除
- `normal`：换人时清除
- `temporary`：按回合结束过期

通常数值含义：

- 攻击/防御：每层 `10%`
- 百分比速度：每层 `10%`
- 固定速度：每层 `10`
- 技能威力百分比：每层 `10%`
- 连击数：每层 `1`
- 能耗：每层 `1`
- buff 数值范围通常限制在 `-99` 到 `99`

特性产生的 buff 使用 `source_kind="trait"`，不计入普通“获得增益”判断。

### 11.2 印记

`sim/marks.py` 中每方有：

- 一个正面印记槽
- 一个负面印记槽

同 ID 印记增加层数；不同 ID 的同类印记会替换旧印记；印记随换人保留。

当前已接入的主要印记：

| ID | 名称 | 效果 |
|---:|---|---|
| `0` | 攻击印记 | 全技能威力 `+10%/层` |
| `1` | 棘刺印记 | 主动换人/脱离时，新入场精灵失去 `6%` 最大生命/层 |
| `2` | 蓄势印记 | 技能威力 `+30%/层`，能耗 `+1/层` |
| `3` | 减速印记 | 速度 `-10/层` |
| `4` | 中毒印记 | 回合结束造成毒系伤害 |
| `5` | 降灵印记 | 主动换人/脱离时，新入场精灵失去 `1` 能量/层 |
| `6` | 蓄电印记 | 入场后首次攻击威力增加 |
| `7` | 星陨印记 | 非幻系攻击触发额外幻系伤害并消耗印记 |
| `8` | 萌芽印记 | 获得增益时额外增加层数 |
| `9` | 龙噬印记 | 使用实际能耗为 3 的技能后获得双攻 |
| `10` | 光合印记 | 回合结束回复能量 |
| `11` | 湿润印记 | 全技能能耗降低 |
| `12` | 风气印记 | 先手攻击时提高本次技能威力 |
| `13` | 暗涌印记 | 主动换人/脱离时，新入场精灵获得随机属性减益 |

### 11.3 天气

`sim/weather.py` 支持：

- 雨天：水系技能威力提升
- 暴风雪：回合结束增加冻结
- 沙暴：部分地系技能能耗降低
- 雷鸣：回合结束增加引电

天气有持续回合数，回合结束递减，到期后清除。

### 11.4 迸发

`sim/burst.py` 中的迸发通常在入场后的首次行动消耗，当前类型包括：

- `attack_power_flat`
- `attack_power_percent`
- `skill_use_count`
- `energy_cost_flat`
- `enemy_energy_cost`

离场时会清除未使用的迸发。

---

## 12. 特性系统

特性系统位于 `sim/traits/`，采用“事件广播 + 数值查询”的双轨设计。

### 12.1 特性绑定

- `BattlePet.trait_id` 来自 `spirits.json` 的 `feature.id`
- `trait_state` 保存特性运行时状态
- 进化、首领化、萌化后通过 `traits.rebind()` 重新绑定特性并清空状态
- 未实现特性由 `registry.py` 注册为 no-op handler

### 12.2 事件广播

引擎在关键时机调用：

```python
traits.emit(state, "事件名", ...)
```

常见事件：

- `turn_start`
- `entry`
- `leave`
- `charge`
- `skill_start`
- `attack`
- `take_damage`
- `lethal`
- `kill`
- `faint`
- `counter`
- `defense`
- `status_skill`
- `skill_end`
- `round_end`
- `buff_gain`
- `energy_gain`
- `dot_damage`
- `heal`
- `weather_change`
- `revive`

默认 `scope="all"`，会广播给双方全部精灵，包括场下精灵。这样可以支持场下积累类特性。

### 12.3 数值查询

主要查询接口：

- `query_stat_multiplier`
- `query_power`
- `query_energy_cost`
- `query_speed`
- `query_damage_dealt`
- `query_damage_taken`
- `query_hit_count`
- `query_lifesteal`
- `query_energy_limit`
- `query_heal`
- `query_energy_gain`
- `query_energy_shortfall`
- `query_skill_usable`
- `query_skill_element`

默认只查询目标精灵自身的特性；能耗和连击查询还会额外查询敌方在场精灵，用于实现敌方光环。

### 12.4 特性可见性

战斗状态不直接暴露特性 ID、特性名称或描述。

已实现特性的部分运行时效果会通过 `trait_effects` 下发，显示为类似：

```text
+双攻 * 3 (20%/trait)
```

---

## 13. 状态序列化与信息隐藏

`battle.state_to_dict(state, view_side=...)` 负责向客户端提供状态。

客户端通常可以看到：

- 双方队伍、生命、能量
- 当前出战位置
- buff/debuff
- 印记
- 天气
- 魔力
- 对方已揭示的技能
- 对方速度范围
- 对方生命百分比

己方会看到：

- 技能真实能耗
- 技能显示威力
- 真实有效速度
- 自己的完整技能信息

对方技能默认不可见，只有使用过的技能才加入 `revealed` 集合并显示。

---

## 14. 当前实现状态和限制

截至当前代码状态，执行：

```bash
python -c "from sim.traits import report; print(report())"
```

特性数据报告为：

```text
total: 231
done: 96
tested: 24
```

这表示：

- `traits.json` 中共有 231 个特性
- 96 个标记为已完成
- 24 个标记为已测试
- 其他特性仍然是 no-op，不产生实际效果

当前的重要边界：

1. **技能效果没有通用执行器。**
   `skills.json` 中的 `effects` 字段尚未由统一运行时解释。许多技能描述中的回血、降能量、永久强化、蓄力等效果不会自动生效。迅捷使用独立的 `BattleSkill.swift` 运行时标记；离场仅作为 `_resolve_action()` 的本次行动结算结果，后续技能效果代码化时应在完成条件判断后设置，不要回退到解析 `desc` 或写入技能 JSON。

2. **状态技能大多仍是占位实现。**
   类别为 `3` 的技能通常只广播 `status_skill` 并记录“效果暂未实现”。

3. **应对技能数据接入不完整。**
   `counter.py` 的判定框架已存在，但 `BattleSkill.counter_target` 默认为空，普通技能构造时尚未完整从数据映射，因此多数应对不会触发。

4. **并非所有 buff 添加点都会广播 `buff_gain`。**
   当前主要在战斗迸发、天气和部分显式接入点广播；特性内部添加 buff 通常不广播，以避免递归连锁。

5. **天气变化事件尚未完全接入。**
   `on_weather_change` 接口已预留，但 `weather.set_weather()` 和 `clear_weather()` 当前主要修改状态并写日志。

6. **首发传动存在边界情况。**
   `create_team_battle()` 会先对默认第 0 只精灵执行传动排序，服务端随后才根据队伍配置修改 `lead`。因此选择非 0 号精灵首发时，首回合的初始传动排序可能不会立即执行。

7. **服务端替补选择在 `step()` 外完成。**
   `step()` 只负责把精灵置为 `active=-1`；死亡替补和特性请求换人由 `server.py` 处理，处理完后才继续战斗。

---

## 15. 开发约束

1. 不要肆意添加输出语句，程序结果应保持精简。
2. 禁止使用 `try-catch` 结构。
3. 不准用模拟数据或伪造数据替代真实结果。
4. `git` 和 `gh` 只允许用于读取信息，不允许提交、推送或创建分支。
5. 修改现有代码前先理解调用链，优先修复根因，不要用表面补丁掩盖状态机问题。
6. 所有回复都以 "gradutate asap" 结尾
