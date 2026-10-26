"""星陨印记相关特性。

引擎星陨段伤害按触发时满层计算；消耗层数经 traits.query_star_consume 查询，
由 marks.consume_mark_stacks 执行（默认全部消耗）。
"""

from __future__ import annotations

from ..base import TraitHandler
from ..registry import register


class StarWatcher(TraitHandler):
    trait_id = 200210
    name = "守望星"
    desc = "触发星陨印记时仅消耗一半层数，仍造成满层伤害。"
    implemented = True

    def modify_star_consume(self, ctx, stacks):
        # 自己触发：消耗 n - floor(n/2)，保留一半（1 层时正常消耗完）
        if ctx.target is not ctx.actor:
            return None
        return stacks - stacks // 2


def register_star() -> None:
    register(StarWatcher())


register_star()
