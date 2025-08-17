"""Triangle module."""

import numpy as np
from numpy.typing import NDArray

from aabb.AABB import AABB
from primitive.primitive import Primitive
from utils.typecheck import is_3d_array
from utils.utils import format_xyz


class Triangle(Primitive):
    """Triangle class."""

    # ---------------------------------- Fields ---------------------------------- #
    _v0: NDArray[np.float32]
    _v1: NDArray[np.float32]
    _v2: NDArray[np.float32]

    # ------------------------------ Dunder methods ------------------------------ #
    def __init__(
        self,
        v0: list[float] | NDArray[np.float32],
        v1: list[float] | NDArray[np.float32],
        v2: list[float] | NDArray[np.float32],
        material: int = 0,
    ) -> None:
        """Initialization method.

        Args:
            v0 (list[float] | NDArray[np.float32]): First vertex coordinates of the triangle
            v1 (list[float] | NDArray[np.float32]): Second vertex coordinates of the triangle
            v2 (list[float] | NDArray[np.float32]): Third vertex coordinates of the triangle
            material (int, optional): Material Index. Defaults to 0.

        """
        self._dtype = np.dtype(
            [
                ("v0", np.float32, 3),  # 12B (4)
                ("material", np.int32),
                ("v1", np.float32, 3),  # 12B (4)
                ("_pad", np.float32),  # 4B  # 8B
                ("v2", np.float32, 3),  # 12B (4)
                ("_pad2", np.float32)
            ],
            align=True,
        )

        self._v0 = None
        self._v1 = None
        self._v2 = None

        super().__init__(material=material)

        self.v0 = v0
        self.v1 = v1
        self.v2 = v2

    def __str__(self) -> str:
        output: str = "Triangle:\n"
        output += f"\tv0: {format_xyz(self.v0)}\n"
        output += f"\tv1: {format_xyz(self.v1)}\n"
        output += f"\tv2: {format_xyz(self.v2)}\n"

        return output

    def __repr__(self) -> str:
        output: str = "Triangle:\n"
        output += f"\tv0: {format_xyz(self.v0)}\n"
        output += f"\tv1: {format_xyz(self.v1)}\n"
        output += f"\tv2: {format_xyz(self.v2)}\n"

        return output

    @property
    def v0(self) -> NDArray[np.float32]:
        """Coordinates of the first vertex.

        Returns:
            NDArray[np.float32]: Array containing the X, Y, Z coordinates of the vertex

        """
        return self._v0

    @v0.setter
    def v0(self, value: NDArray[np.float32] | list[float]) -> None:
        self._v0 = is_3d_array(value)

        self._calculate_aabb()
        self._calculate_ogl_ssbo_array()

    @property
    def v1(self) -> NDArray[np.float32]:
        """Coordinates of the second vertex.

        Returns:
            NDArray[np.float32]: Array containing the X, Y, Z coordinates of the vertex

        """
        return self._v1

    @v1.setter
    def v1(self, value: NDArray[np.float32] | list[float]) -> None:
        self._v1 = is_3d_array(value)

        self._calculate_aabb()
        self._calculate_ogl_ssbo_array()

    @property
    def v2(self) -> NDArray[np.float32]:
        """Coordinates of the third vertex.

        Returns:
            NDArray[np.float32]: Array containing the X, Y, Z coordinates of the vertex

        """
        return self._v2

    @v2.setter
    def v2(self, value: NDArray[np.float32] | list[float]) -> None:
        self._v2 = is_3d_array(value)

        self._calculate_aabb()
        self._calculate_ogl_ssbo_array()

    # ------------------------------ Private methods ----------------------------- #
    def _calculate_aabb(self) -> None:
        # if the properties aren't initialized, skip the execution
        if self._v0 is None or self._v1 is None or self._v2 is None:
            return

        # Stack into shape (3, 3) → 3 vertices, each with (x, y, z)
        verts = np.stack([self._v0, self._v1, self._v2])

        # Compute min and max per axis
        min_coords: NDArray[np.float32] = np.min(verts, axis=0)
        max_coords: NDArray[np.float32] = np.max(verts, axis=0)

        self._aabb = AABB(min=min_coords, max=max_coords)

    def _calculate_ogl_ssbo_array(self) -> None:
        """Private method for calculating the data array for OpenGL SSBO."""
        # if the properties aren't initialized, skip the execution
        if self._v0 is None or self._v1 is None or self._v2 is None:
            return

        # reset the data array
        self._ogl_ssbo_data = np.zeros(1, dtype=self._dtype)

        # add the values to the array with the specified structure
        self._ogl_ssbo_data[0]["v0"] = [self._v0[0], self._v0[1], self._v0[2]]
        self._ogl_ssbo_data[0]["v1"] = [self._v1[0], self._v1[1], self._v1[2]]
        self._ogl_ssbo_data[0]["v2"] = [self._v2[0], self._v2[1], self._v2[2]]
        self._ogl_ssbo_data[0]["material"] = self._material
        self._ogl_ssbo_data[0]["_pad"] = 0.0  # padding float
        self._ogl_ssbo_data[0]["_pad2"] = 0.0  # padding float
