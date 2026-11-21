from __future__ import annotations

import random as _random
import threading

import sim.actions as _sim_actions
import sim.battle as _sim_battle
import sim.offerings as _sim_offerings
import sim.skills.ops as _sim_skill_ops
import sim.traits as _sim_traits
import sim.traits.impl.power as _sim_trait_power
import sim.traits.impl.resource as _sim_trait_resource


class ThreadRandom:
    def __init__(self):
        self.local = threading.local()

    def _rng(self):
        if not hasattr(self.local, "rng"):
            self.local.rng = _random.Random()
        return self.local.rng

    def seed(self, value):
        self._rng().seed(value)

    def __getattr__(self, name):
        return getattr(self._rng(), name)


THREAD_RANDOM = ThreadRandom()
for _module in (
    _sim_actions,
    _sim_battle,
    _sim_offerings,
    _sim_skill_ops,
    _sim_traits,
    _sim_trait_power,
    _sim_trait_resource,
):
    _module.random = THREAD_RANDOM
