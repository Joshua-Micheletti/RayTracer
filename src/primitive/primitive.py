"""primitive module."""

from abc import ABC, abstractmethod

from aabb import AABB
from core import SSBOData


class Primitive(SSBOData, ABC):
    """Abstract class to create primitives for rendering."""

    _aabb: AABB
    _material: int

    def __init__(self, material: int = 0) -> None:
        """
        Initialization method.

        Args:
            material (int, optional): Material index. Defaults to 0.

        """
        self._material = material
        super().__init__()

    @property
    def aabb(self) -> AABB:
        """
        Axis alligned bounding box of the sphere.

        Returns:
            AABB: AABB object containing the extremes of the sphere

        """
        return self._aabb

    @property
    def material(self) -> int:
        """
        Index to the material list to use to render the sphere.

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
