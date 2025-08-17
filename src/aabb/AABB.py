import numpy as np
from numpy.typing import NDArray

from utils.utils import format_xyz


class AABB:
    def __init__(self, min: NDArray[np.float32] = [], max: NDArray[np.float32] = []) -> None:
        self.min: NDArray[np.float32] = min
        self.max: NDArray[np.float32] = max

    def __str__(self) -> str:
        output: str = "AABB:\n"

        output += f"\tMax: {format_xyz(self.max)}\n"
        output += f"\tMin: {format_xyz(self.min)}"

        return output

    def __repr__(self) -> str:
        output: str = "AABB: \n"

        output += f"\tMax: {format_xyz(self.max)}\n"
        output += f"\tMin: {format_xyz(self.min)}"

        return output
