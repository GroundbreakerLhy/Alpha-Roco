"""Shared enums and simple constants."""

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


class SkillCategory(IntEnum):
    PHYSICAL = 0
    MAGIC = 1
    DEFENSE = 2
    STATUS = 3


# 血脉编号：0-17 为元素血脉（与 Element 一致，见 spirits.json bloodline）；
# 首领血脉单独编号，接在元素之后。
LORD_BLOODLINE = 18


ACTION_KINDS = ("skill", "charge")
