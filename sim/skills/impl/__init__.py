"""技能 handler 实现包：按类别分文件，每个文件内自注册。

简单技能用 ``ops.OpsSkill`` 声明效果原语，复杂技能继承 ``SkillHandler`` 覆写钩子。
"""

from __future__ import annotations

from . import defense  # noqa: F401
from . import attack  # noqa: F401
from . import status  # noqa: F401
from . import wish  # noqa: F401
