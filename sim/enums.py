"""共享枚举与常量。"""

from enum import IntEnum


class Element(IntEnum):
    NORMAL = 0
    GRASS = 1
    FIRE = 2
    WATER = 3
    LIGHT = 4
    EARTH = 5
    ICE = 6
    DRAGON = 7
    ELECTRIC = 8
    POISON = 9
    BUG = 10
    FIGHT = 11
    WING = 12
    CUTE = 13
    GHOST = 14
    DARK = 15
    MECH = 16
    FANTASY = 17


ELEMENT_NAMES = (
    "普通", "草", "火", "水", "光", "地", "冰", "龙",
    "电", "毒", "虫", "武", "翼", "萌", "幽", "恶", "机械", "幻",
)


class SkillCategory(IntEnum):
    PHYSICAL = 0
    MAGIC = 1
    DEFENSE = 2
    STATUS = 3


# 血脉：0-17 元素（同 Element）；18=首领。
LORD_BLOODLINE = 18


ACTION_KINDS = ("skill", "charge")
