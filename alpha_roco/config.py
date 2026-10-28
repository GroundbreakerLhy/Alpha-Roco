from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ModelConfig:
    hidden_dim: int = 384  # 隐藏维度
    num_heads: int = 8  # 注意力头数
    num_layers: int = 3  # Attention 层数
    ffn_multiplier: int = 4  # FFN 隐藏维度倍数
    gru_layers: int = 2  # GRU 层数
    identity_embedding_dim: int = 32  # 身份嵌入维度
    category_embedding_dim: int = 16  # 类别/属性嵌入维度
    dropout: float = 0.1  # dropout 比例

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        return cls(**values)


@dataclass
class RewardConfig:
    team_size: int = 6  # 队伍精灵数
    hp_potential: float = 0.5  # 血量势能系数
    alive_potential: float = 0.1  # 存活势能系数
    effect_potential: float = 0.02  # 强化/减益势能系数
    effect_clip: float = 2.0  # 效果项截断范围
    lead_oneshot_bonus: float = 0.05  # 首发可一击击杀对方的奖励
    lead_survive_bonus: float = 0.05  # 首发不被对方一击击杀的奖励
    rare_skill_cap: float | None = None  # 激进探索阶段单局上限；None 表示不限制
    rare_skill_cap_late: float = 0.10  # 后续探索阶段单局上限
    trait_novelty_bonus: float = 0.05  # 首次触发某类特性的探索奖励
    switch_first_turn_penalty: float = 0.05  # 第一回合主动换人惩罚
    switch_consecutive_penalty: float = 0.1  # 连续主动换人惩罚


@dataclass
class TeacherConfig:
    lead_attack: float = 1.0  # 首发进攻克制权重
    lead_defense: float = 1.0  # 首发防守克制权重
    lead_speed: float = 0.1  # 首发速度权重
    lead_hp: float = 0.1  # 首发血量权重
    replacement_offense: float = 0.5  # 补位进攻克制权重
    replacement_defense: float = -0.5  # 补位防守克制权重
    replacement_hp: float = 0.3  # 补位血量权重
    replacement_speed: float = 0.2  # 补位速度权重
    danger_low_hp: float = 0.2  # 低血量危险阈值
    danger_mid_hp: float = 0.35  # 中血量危险阈值
    danger_incoming: float = 0.5  # 中血量时预计承伤危险阈值
    danger_lethal: float = 1.0  # 预计承伤致命阈值
    skill_damage: float = 1.0  # 技能预计伤害权重
    skill_type_advantage: float = 0.1  # 克制倍率优势权重
    skill_type_disadvantage: float = -0.05  # 被抵抗惩罚权重
    skill_counter_flags: float = 0.02  # 技能应对类别标记权重
    skill_speed: float = 0.05  # 先手权重
    skill_drive: float = 0.03  # 传动权重
    skill_windup: float = -0.05  # 蓄力惩罚
    counter_attack: float = 0.30  # 应对攻击奖励
    counter_defense: float = 0.25  # 应对防御奖励
    counter_status: float = 0.25  # 应对状态奖励
    counter_interrupt: float = 0.05  # 打断奖励
    defense_danger: float = 0.1  # 危险时防御技能奖励
    kill_threshold: float = 0.1  # 一击击杀回合阈值
    kill_bonus: float = 1.0  # 一击击杀奖励
    switch_damage: float = 0.4  # 换人后伤害权重
    switch_incoming: float = -0.35  # 换人承伤惩罚
    switch_hp: float = 0.1  # 换人血量权重
    switch_speed: float = 0.05  # 换人速度权重
    switch_offense: float = 0.2  # 换人进攻克制权重
    switch_defense: float = -0.2  # 换人防守克制权重
    switch_lethal: float = -0.25  # 换人后被一击击杀惩罚
    switch_lethal_threshold: float = 1.0  # 换人承伤致命阈值
    switch_danger: float = 0.25  # 危险时换人奖励
    switch_resisted: float = 0.15  # 技能被抵抗时换人奖励
    switch_low_hp: float = -0.2  # 低血量换人惩罚
    switch_low_energy: float = -0.2  # 低能量换人惩罚
    switch_switched_last: float = -0.2  # 连续换人惩罚
    switch_threshold: float = 0.05  # 换人分数超过最佳技能的阈值
    switch_offensive_margin: float = 0.15  # 换人进攻伤害差阈值
    switch_offensive_fraction: float = 0.35  # 换人进攻伤害绝对值阈值
    switch_offensive_current: float = 0.15  # 当前技能伤害上限
    switch_hp_threshold: float = 0.25  # 换人低血量阈值
    switch_energy_threshold: float = 2  # 换人低能量阈值
    charge_no_usable: float = 0.5  # 无可用技能时聚能分数
    charge_low_energy: float = 0.15  # 低能量时聚能分数
    charge_energy_threshold: float = 4  # 低能量阈值
    default_energy: float = 10  # 未知能量默认值


@dataclass
class SearchConfig:
    top_k: int = 5  # 搜索取对手候选数
    risk: float = 0.5  # 交互分数风险权重


@dataclass
class EvaluationConfig:
    games: int = 256  # 评测局数
    workers: int = 32  # 评测并行数
    timeout: float = 20.0  # 单局超时秒数
    seed: int = 0  # 评测种子


@dataclass
class PretrainConfig:
    games: int = 256  # 预训练 teacher 对局数
    epochs: int = 2  # BC epoch 数
    workers: int = 16  # 采集并行数
    chunk_size: int = 64  # 每块采集局数
    timeout: float = 20.0  # 单局超时秒数
    learning_rate: float = 1e-4  # BC 学习率
    seed: int = 0  # 预训练种子
    output: str = "logs/pretrain"  # 预训练输出目录


@dataclass
class TrainConfig:
    batches: int = 2  # PPO 批次数
    games_per_batch: int = 56  # 每批对局数
    ppo_epochs: int = 2  # 每批 PPO epoch 数
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # 训练设备
    seed: int = 0  # 训练种子
    output: str = "logs/az"  # 输出目录
    learning_rate: float = 5e-5  # 学习率
    clip_ratio: float = 0.2  # PPO 裁剪比例
    value_coefficient: float = 0.5  # 价值损失系数
    entropy_coefficient: float = 0.03  # 熵奖励系数
    gamma: float = 1.0  # 折扣因子
    gae_lambda: float = 0.95  # GAE 平滑系数
    max_grad_norm: float = 0.5  # 梯度裁剪上限
    target_kl: float = 0.02  # KL 早停阈值
    threads: int = 1  # PyTorch 线程数
    log_games: bool = False  # 是否记录对局日志
    keep_checkpoints: int = 1  # 保留 checkpoint 数
    pool_size: int = 8  # best 池大小
    eval_interval: int = 8  # 评测间隔批数
    eval_games: int = 256  # 每次评测局数
    eval_workers: int = 32  # 评测并行数
    sample_workers: int = 16  # 采样并行数
    update_chunk: int = 16  # 更新分块局数
    terminal_timeout: float = 20.0  # 单局超时秒数
    opponent_current_prob: float = 0.5  # 对手为当前 best 的概率
    pool_rate_low: float = 0.2  # 池内优先胜率下界
    pool_rate_high: float = 0.8  # 池内优先胜率上界
    pool_weight: float = 0.2  # 池内非优先权重
    promotion_threshold: float = 0.55  # 晋升胜率阈值
    elo_k: float = 32.0  # Elo 更新系数
    aux_coefficient: float = 0.1  # 辅助损失系数
    effect_aux_coefficient: float = 0.1  # 状态变化/效果差辅助损失系数
    trait_aux_coefficient: float = 0.1  # 特性触发/效果辅助损失系数
    state_change_pos_weight: float = 10.0  # 状态变化 BCE 正例权重上限
    rare_skill_bonus: float = 0.08  # 激进探索阶段技能奖励系数
    rare_skill_bonus_late: float = 0.02  # 后续阶段技能奖励系数
    exploration_aggressive_batches: int = 0  # 激进探索批次数；0 表示直接使用后续阶段
    status_exploration_weight: float = 2.5  # 激进阶段状态技能权重
    status_exploration_weight_late: float = 1.2  # 后续阶段状态技能权重
    rare_skill_batches: int = 0  # 技能探索批次数；0 表示全程
    reward: RewardConfig = field(default_factory=RewardConfig)  # 奖励配置
