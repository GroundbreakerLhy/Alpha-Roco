"""属性克制倍率：效果拔群合并为 1+层数，其余倍率相乘。"""

from .data_loader import load_json

_MATRIX = None


def type_multiplier(
    attack_element: int, defender_elements: list, typechart: dict | None = None
) -> float:
    if not defender_elements:
        return 1.0
    global _MATRIX
    if typechart and "matrix" in typechart:
        matrix = typechart["matrix"]
    else:
        if _MATRIX is None:
            _MATRIX = load_json("typechart.json")["matrix"]
        matrix = _MATRIX
    row = matrix[attack_element]
    supers = 0
    other = 1.0
    for element in defender_elements:
        value = row[element]
        if value == 2:
            supers += 1
        else:
            other *= value
    return (1 + supers if supers else 1) * other
