"""Triangle module."""

import numpy as np
from numpy.typing import NDArray

from aabb import AABB
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
        v0: list[float] | NDArray[np.float32] = None,
        v1: list[float] | NDArray[np.float32] = None,
        v2: list[float] | NDArray[np.float32] = None,
        material: int = 0,
    ) -> None:
        """
        Initialization method.

        Args:
            v0 (list[float] | NDArray[np.float32]): First vertex coordinates of the triangle. Defaults to [-0.5, 0.0, -0.5]
            v1 (list[float] | NDArray[np.float32]): Second vertex coordinates of the triangle. Defaults to [0.0, 1.0, 0.0]
            v2 (list[float] | NDArray[np.float32]): Third vertex coordinates of the triangle. Defaults to [0.5, 0.0, 0.5]
            material (int, optional): Material Index. Defaults to 0.

        """  # noqa: E501
        if v0 is None:
            v0 = [-0.5, 0.0, -0.5]
        if v1 is None:
            v1 = [0.0, 1.0, 0.0]
        if v2 is None:
            v2 = [0.5, 0.0, 0.5]

        f4 = np.dtype("<f4")  # little-endian float32
        u4 = np.dtype("<u4")  # little-endian uint32

        self._dtype = np.dtype(
            {
                "names": ["v0", "material", "v1", "_pad1", "v2", "_pad2", "normal", "_pad3"],
                "formats": [(f4, 3), u4, (f4, 3), f4, (f4, 3), f4, (f4, 3), f4],
                "offsets": [0, 12, 16, 28, 32, 44, 48, 60],
                "itemsize": 64,  # total struct size = 64 bytes (multiple of 16)
            }
        )

        self._v0 = None
        self._v1 = None
        self._v2 = None

        super().__init__(material=material)

        self.v0 = v0
        self.v1 = v1
        self.v2 = v2

    def __str__(self) -> str:
        """
        Dunder method for representing the object as a string.

        Returns:
            str: Formatted string

        """
        output: str = "Triangle:\n"
        output += f"\tv0: {format_xyz(self.v0)}\n"
        output += f"\tv1: {format_xyz(self.v1)}\n"
        output += f"\tv2: {format_xyz(self.v2)}\n"

        return output

    def __repr__(self) -> str:
        """
        Dunder method for representing the object as a string.

        Returns:
            str: Formatted string

        """
        output: str = "Triangle:\n"
        output += f"\tv0: {format_xyz(self.v0)}\n"
        output += f"\tv1: {format_xyz(self.v1)}\n"
        output += f"\tv2: {format_xyz(self.v2)}\n"

        return output

    @property
    def v0(self) -> NDArray[np.float32]:
        """
        Coordinates of the first vertex.

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
        """
        Coordinates of the second vertex.

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
        """
        Coordinates of the third vertex.

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

        self._aabb = AABB(min_coords=min_coords, max_coords=max_coords)

    def _calculate_ogl_ssbo_array(self) -> None:
        """Private method for calculating the data array for OpenGL SSBO."""
        # if the properties aren't initialized, skip the execution
        if self._v0 is None or self._v1 is None or self._v2 is None:
            return

        # Compute edge vectors
        edge01 = self._v1 - self._v0
        edge02 = self._v2 - self._v0

        # Compute the normal
        normal = np.cross(edge01, edge02)
        normal = normal / np.linalg.norm(normal)  # normalize

        # reset the data array
        self._ogl_ssbo_data = np.zeros(1, dtype=self._dtype)

        # add the values to the array with the specified structure
        self._ogl_ssbo_data[0]["v0"] = [self._v0[0], self._v0[1], self._v0[2]]
        self._ogl_ssbo_data[0]["v1"] = [self._v1[0], self._v1[1], self._v1[2]]
        self._ogl_ssbo_data[0]["v2"] = [self._v2[0], self._v2[1], self._v2[2]]
        self._ogl_ssbo_data[0]["material"] = self._material
        self._ogl_ssbo_data[0]["normal"] = [normal[0], normal[1], normal[2]]
        # self._ogl_ssbo_data[0]["edge1"] = edge01
        # self._ogl_ssbo_data[0]["edge2"] = edge02
        # self._ogl_ssbo_data[0]["_pad"] = 0.0  # padding float
        # self._ogl_ssbo_data[0]["_pad2"] = 0.0  # padding float
        # self._ogl_ssbo_data[0]["_pad3"] = 0.0
        # self._ogl_ssbo_data[0]["_pad4"] = 0.0
        # self._ogl_ssbo_data[0]["_pad5"] = 0.0
