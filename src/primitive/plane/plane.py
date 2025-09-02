"""Plane module."""

import numpy as np
from numpy.typing import NDArray

from primitive.primitive import Primitive
from utils.typecheck import is_3d_array


class Plane(Primitive):
    """Plane class."""

    _point: NDArray[np.float32]
    _normal: NDArray[np.float32]

    # ------------------------------ Dunder methods ------------------------------ #
    def __init__(
        self,
        point: list[float] | NDArray[np.float32] = None,
        normal: list[float] | NDArray[np.float32] = None,
        material: int = 0,
    ) -> None:
        """
        Initialization method.

        Args:
            point (list[float] | NDArray[np.float32]): Point that lies on the plane. Defaults to [0.0, 0.0, 0.0]
            normal (list[float] | NDArray[np.float32]): Normal vector of the plane. Defaults to [0.0, 1.0, 0.0]
            material (int): Material index. Defaults to 0

        """  # noqa: E501
        if point is None:
            point = [0.0, 0.0, 0.0]
        if normal is None:
            normal = [0.0, 1.0, 0.0]

        self._dtype = np.dtype(
            [
                ("point", np.float32, 3),  # 12B
                ("material", np.uint32),  # 4B
                ("normal", np.float32, 3),  # 12B
                ("_pad", np.int32),  # 4B
            ],
            align=True,
        )

        # create the private fields
        self._point = None
        self._normal = None

        super().__init__(material=material)

        self.point = point
        self.normal = normal

    # ---------------------------- Setters and Getters --------------------------- #
    @property
    def point(self) -> NDArray[np.float32]:
        """
        Coordinates of a point that lies on the plane.

        Returns:
            NDArray[np.float32]: Array containing the X, Y, Z coordinates of the point

        """
        return self._point

    @point.setter
    def point(self, value: NDArray[np.float32] | list[float]) -> None:
        self._point = is_3d_array(value)

        self._calculate_ogl_ssbo_array()

    @property
    def normal(self) -> NDArray[np.float32]:
        """
        Normal vector of the plane.

        Returns:
            NDArray[np.float32]: Array containing the X, Y, Z components of the normal vector

        """
        return self._normal

    @normal.setter
    def normal(self, value: NDArray[np.float32] | list[float]) -> None:
        self._normal = is_3d_array(value)

        self._calculate_ogl_ssbo_array()

    # ------------------------------ Private methods ----------------------------- #
    def _calculate_aabb(self) -> None:
        """Unused method as the plane has no AABB."""

    def _calculate_ogl_ssbo_array(self) -> None:
        """Private method for calculating the data array for OpenGL SSBO."""
        # if the properties of the sphere aren't initialized, skip the execution
        if self._point is None or self._normal is None:
            return
        # reset the data array
        self._ogl_ssbo_data = np.zeros(1, dtype=self._dtype)

        # add the values to the array with the specified structure
        self._ogl_ssbo_data[0]["point"] = [self._point[0], self._point[1], self._point[2]]
        self._ogl_ssbo_data[0]["material"] = self._material
        self._ogl_ssbo_data[0]["normal"] = [self._normal[0], self._normal[1], self._normal[2]]
        self._ogl_ssbo_data[0]["_pad"] = 0.0  # padding float
