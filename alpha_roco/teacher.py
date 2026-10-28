from __future__ import annotations

from .config import TeacherConfig
from .features import ObservationEncoder, selectable_candidates


class Teacher:
    def __init__(self, encoder=None, seed=0, config=None):
        self.encoder = encoder or ObservationEncoder()
        self.seed = seed
        self.config = config or TeacherConfig()

    def choose(self, request):
        phase = request.get("phase", "action")
        if phase == "lead":
            return self._lead(request)
        candidates = selectable_candidates(request)
        encoded = self.encoder.encode(request)["own_candidates"]
        scores = (
            self._replacement_scores(encoded)
            if phase == "replacement"
            else self._action_scores(request, candidates, encoded)
        )
        return candidates[max(range(len(candidates)), key=scores.__getitem__)]

    def _lead(self, request):
        team = request.get("team", [])
        enemies = self._opponents(request)
        speeds = [self._speed(raw) for raw in team]
        top_speed = max(speeds, default=0.0) or 1.0
        candidates = selectable_candidates(request)
        best = None
        for candidate in candidates:
            index = int(candidate.get("lead", candidate.get("pet_index", -1)))
            if not 0 <= index < len(team):
                continue
            attack = (
                sum(
                    self.encoder._best_mult(team[index], self.encoder._elements(enemy))
                    for enemy in enemies
                )
                / len(enemies)
                if enemies
                else 1.0
            )
            defense = (
                sum(
                    self.encoder._best_mult(enemy, self.encoder._elements(team[index]))
                    for enemy in enemies
                )
                / len(enemies)
                if enemies
                else 1.0
            )
            hp = (
                self.encoder._hp_ratio(team[index])
                if "hp" in team[index] or "hp_percent" in team[index]
                else 1.0
            )
            score = (
                self.config.lead_attack * attack
                - self.config.lead_defense * defense
                + self.config.lead_speed * speeds[index] / top_speed
                + self.config.lead_hp * hp
            )
            if best is None or score > best[0]:
                best = (score, candidate)
        return best[1] if best is not None else candidates[0]

    def _replacement_scores(self, encoded):
        return [
            item["derived"][16] * self.config.replacement_offense
            + item["derived"][17] * self.config.replacement_defense
            + item["derived"][18] * self.config.replacement_hp
            + item["derived"][19] * self.config.replacement_speed
            for item in encoded
        ]

    def _action_scores(self, request, candidates, encoded):
        state = request.get("state", {})
        side = request.get("side", "A")
        enemy_side = "B" if side == "A" else "A"
        own = state.get("teams", {}).get(side, [])
        current = self._active(request, side)
        enemy = self._active(request, enemy_side)
        current_hp = (
            self.encoder._hp_ratio(current)
            if "hp" in current or "hp_percent" in current
            else 0.0
        )
        current_hp_value = current.get("hp")
        if not isinstance(current_hp_value, (int, float)):
            current_hp_value = current_hp * current.get("max_hp", 100)
        incoming = self._best_damage(enemy, current) if enemy and current else 0
        incoming_fraction = incoming / max(current_hp_value, 1)
        danger = (
            current_hp < self.config.danger_low_hp
            or (
                current_hp < self.config.danger_mid_hp
                and incoming_fraction >= self.config.danger_incoming
            )
            or incoming_fraction >= self.config.danger_lethal
        )
        energy = current.get("energy")
        energy = (
            energy
            if isinstance(energy, (int, float)) and not isinstance(energy, bool)
            else 0.0
        )
        predicted = self._last_action_category(request, enemy_side)
        switched_last = self._last_switched(request, side)
        enemy_hp = self.encoder._current_hp(enemy, self.encoder._stats(enemy))

        scores = [0.0] * len(candidates)
        usable = []
        for index, candidate in enumerate(candidates):
            if candidate.get("kind") != "skill":
                continue
            derived = encoded[index]["derived"]
            if derived[6] <= 0.5 or derived[14] <= 0.5:
                continue
            mult = 2 ** (derived[0] * 2)
            counter = (
                "attack"
                if derived[7] > 0.5
                else (
                    "defense"
                    if derived[8] > 0.5
                    else "status" if derived[9] > 0.5 else ""
                )
            )
            score = (
                self.config.skill_damage * derived[2]
                + self.config.skill_type_advantage * max(0.0, mult - 1)
                + self.config.skill_type_disadvantage * max(0.0, 1 - mult)
                + self.config.skill_counter_flags * sum(derived[7:10])
                + self.config.skill_speed * derived[4]
                + self.config.skill_drive * derived[11]
                + self.config.skill_windup * derived[13]
            )
            if counter and counter == predicted:
                score += {
                    "attack": self.config.counter_attack,
                    "defense": self.config.counter_defense,
                    "status": self.config.counter_status,
                }[counter]
                if derived[10] > 0.5:
                    score += self.config.counter_interrupt
            if (
                self.encoder._skill(candidate.get("skill_id"))["category_raw"] == 2
                and danger
            ):
                score += self.config.defense_danger
            if 0 < derived[3] <= self.config.kill_threshold:
                score += self.config.kill_bonus
            scores[index] = score
            usable.append(
                {
                    "score": score,
                    "mult": mult,
                    "fraction": min(derived[2] * 2, 1.0),
                    "hits": derived[3],
                    "speed": derived[4],
                }
            )

        best_score = max((item["score"] for item in usable), default=0.0)
        current_fraction = max((item["fraction"] for item in usable), default=0.0)
        damaging = [item["mult"] for item in usable if item["fraction"] > 0]
        resisted = bool(damaging) and max(damaging) < 1.0
        finish_fast = any(
            0 < item["hits"] <= self.config.kill_threshold and item["speed"] > 0.5
            for item in usable
        )
        one_shot = any(
            0 < item["hits"] <= self.config.kill_threshold for item in usable
        )

        for index, candidate in enumerate(candidates):
            if candidate.get("kind") != "switch":
                continue
            off, defense, hp, speed = encoded[index]["derived"][16:20]
            pet_index = int(candidate.get("pet_index", -1))
            bench = own[pet_index] if 0 <= pet_index < len(own) else {}
            bench_energy = bench.get("energy", 10)
            bench_damage = self._best_damage(bench, enemy)
            bench_fraction = min(bench_damage / max(enemy_hp, 1), 1.0)
            bench_hp = self.encoder._current_hp(bench, self.encoder._stats(bench))
            bench_incoming = min(
                self._best_damage(enemy, bench) / max(bench_hp, 1), 1.0
            )
            offensive = (
                bench_fraction
                >= current_fraction + self.config.switch_offensive_margin
                or (
                    bench_fraction >= self.config.switch_offensive_fraction
                    and current_fraction <= self.config.switch_offensive_current
                )
            )
            if finish_fast or (switched_last and not danger):
                scores[index] = -1.0
                continue
            if not (danger or resisted or offensive):
                scores[index] = -1.0
                continue
            score = (
                self.config.switch_damage * bench_fraction
                + self.config.switch_incoming * bench_incoming
                + self.config.switch_hp * hp
                + self.config.switch_speed * speed
                + self.config.switch_offense * off
                + self.config.switch_defense * defense
            )
            if bench_incoming >= self.config.switch_lethal_threshold:
                score += self.config.switch_lethal
            if danger:
                score += self.config.switch_danger
            if resisted:
                score += self.config.switch_resisted
            if hp < self.config.switch_hp_threshold:
                score += self.config.switch_low_hp
            if (
                isinstance(bench_energy, (int, float))
                and bench_energy < self.config.switch_energy_threshold
            ):
                score += self.config.switch_low_energy
            if switched_last:
                score += self.config.switch_switched_last
            scores[index] = score
        if usable and not (danger or resisted):
            threshold = best_score + self.config.switch_threshold
            for index, candidate in enumerate(candidates):
                if candidate.get("kind") == "switch" and scores[index] < threshold:
                    scores[index] = -1.0

        charge = (
            self.config.charge_no_usable
            if not usable
            else (
                self.config.charge_low_energy
                if energy < self.config.charge_energy_threshold and not one_shot
                else 0.0
            )
        )
        for index, candidate in enumerate(candidates):
            if candidate.get("kind") == "charge":
                scores[index] = charge
        return scores

    def _active(self, request, side):
        state = request.get("state", {})
        index = state.get("active", {}).get(side, -1)
        team = state.get("teams", {}).get(side, [])
        return team[index] if 0 <= index < len(team) else {}

    def _skill_list(self, raw):
        skills = raw.get("skills") or self.encoder._species(raw).get("skills", {}).get(
            "normal", []
        )
        return [self.encoder._skill(skill) for skill in skills]

    def _best_damage(self, attacker, defender):
        energy = attacker.get("energy")
        if not isinstance(energy, (int, float)) or isinstance(energy, bool):
            energy = self.config.default_energy
        return max(
            (
                self.encoder._estimated_damage(skill, attacker, defender)
                for skill in self._skill_list(attacker)
                if not isinstance(skill["energy_raw"], (int, float))
                or skill["energy_raw"] <= energy
            ),
            default=0,
        )

    def _last_action_line(self, request, side):
        events = list(request.get("events", []))
        if not events:
            events = list((request.get("public_state") or {}).get("log", []))
        for event in reversed(events):
            line = event if isinstance(event, str) else event.get("line", "")
            text = line.lstrip()
            if text.startswith(f"{side} "):
                return text
        return ""

    def _last_action_category(self, request, side):
        line = self._last_action_line(request, side)
        if "聚能" in line:
            return "status"
        if "使用 " in line:
            name = line.split("使用 ", 1)[1].strip()
            item = self.encoder.skill_by_name.get(name)
            category = item.get("category") if item else None
            return {0: "attack", 1: "attack", 2: "defense", 3: "status"}.get(
                category, ""
            )
        return ""

    def _last_switched(self, request, side):
        return "换上 " in self._last_action_line(request, side)

    def _opponents(self, request):
        state = request.get("state", {})
        enemy_side = "B" if request.get("side", "A") == "A" else "A"
        return state.get("teams", {}).get(enemy_side) or [
            {"name": name} for name in request.get("opponent_team") or []
        ]

    def _speed(self, raw):
        speed = self.encoder._speed(raw)
        if speed is not None:
            return speed
        return float(self.encoder._species(raw).get("stats", {}).get("speed", 0.0))
