from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from aabb.AABB import AABB


class Primitive(ABC):
    _aabb: AABB
    _material: int
    _dtype: np.dtype
    _ogl_ssbo_data: NDArray

    def __init__(self, material: int = 0) -> None:
        self._material = material
        self._ogl_ssbo_data = np.zeros(1, dtype=self.dtype)

    @property
    def aabb(self) -> AABB:
        """Axis alligned bounding box of the sphere.

        Returns:
            AABB: AABB object containing the extremes of the sphere

        """
        return self._aabb

    @property
    def dtype(self) -> np.dtype:
        """Numpy dtype for the formatted data of the sphere.

        Returns:
            np.dtype: Type of the sphere data

        """
        return self._dtype

    @property
    def ogl_ssbo_data(self) -> NDArray:
        """Data representing the primitive formatted to STD430 OpenGL SSBO standard layout.

        Returns:
            NDarray[self._dtype]: Formatted array

        """
        return self._ogl_ssbo_data

    @property
    def material(self) -> int:
        """Index to the material list to use to render the sphere.

        Returns:
            int: Material index

        """
        return self._material

    @material.setter
    def material(self, value: int) -> None:
        if not isinstance(value, int):
            message: str = "Material is an index and should be an integer"
            raise TypeError(message)

        self._material = value
        self._calculate_ogl_ssbo_array()

    @abstractmethod
    def _calculate_aabb(self) -> None:
        pass

    @abstractmethod
    def _calculate_ogl_ssbo_array(self) -> None:
        pass
