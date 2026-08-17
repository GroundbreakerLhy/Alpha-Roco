"""Type effectiveness lookup."""


def type_multiplier(attack_element: int, defender_elements: list, typechart: dict) -> float:
    if not defender_elements:
        return 1.0
    if len(defender_elements) == 1:
        table = typechart["single"].get(str(defender_elements[0]))
        if not table:
            return 1.0
        mult = 1.0
        for entry in table.get("weak", []):
            if entry["type"] == attack_element:
                mult *= entry["value"]
        for entry in table.get("vulnerable", []):
            if entry["type"] == attack_element:
                mult *= entry["value"]
        return mult

    a, b = sorted(defender_elements[:2])
    table = typechart["dual"].get(f"{a}-{b}")
    if not table:
        return 1.0
    mult = 1.0
    for entry in table.get("weak", []):
        if entry["type"] == attack_element:
            mult *= entry["value"]
    for entry in table.get("vulnerable", []):
        if entry["type"] == attack_element:
            mult *= entry["value"]
    return mult
