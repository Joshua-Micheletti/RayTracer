"""aabb module."""

import numpy as np
from numpy.typing import NDArray

from utils.utils import format_xyz


class AABB:
    """aabb class."""

    def __init__(
        self, min_coords: NDArray[np.float32] = None, max_coords: NDArray[np.float32] = None
    ) -> None:
        """
        Initialization method.

        Args:
            min_coords (NDArray[np.float32], optional): minimum coordinates. Defaults to [].
            max_coords (NDArray[np.float32], optional): maximum coordinates. Defaults to [].

        """
        if min_coords is None:
            min_coords = []
        if max_coords is None:
            max_coords = []

        self.min: NDArray[np.float32] = min_coords
        self.max: NDArray[np.float32] = max_coords

    def __str__(self) -> str:
        """
        Method for representing the object as a string.

        Returns:
            str: Formatted string

        """
        output: str = "AABB:\n"

        output += f"\tMax: {format_xyz(self.max)}\n"
        output += f"\tMin: {format_xyz(self.min)}"

        return output

    def __repr__(self) -> str:
        """
        Method for representing the object as a string.

        Returns:
            str: Formatted string

        """
        output: str = "AABB: \n"

        output += f"\tMax: {format_xyz(self.max)}\n"
        output += f"\tMin: {format_xyz(self.min)}"

        return output

    def surface_area(self) -> float:
        """
        Method for calculating the surface area of the bounding box.

        Returns:
            float: Surface area

        """
        d = self.max - self.min
        return 2.0 * (d[0] * d[1] + d[0] * d[2] + d[1] * d[2])

    def union(self, other: "AABB") -> "AABB":
        """
        Method for joining AABBs.

        Args:
            other (AABB): Other AABB to join with

        Returns:
            AABB: Conjoined AABB

        """
        new_min = np.minimum(self.min, other.min)
        new_max = np.maximum(self.max, other.max)
        return AABB(new_min, new_max)
