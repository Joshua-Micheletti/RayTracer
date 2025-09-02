"""Module to represent a sphere."""

import numpy as np
from numpy.typing import NDArray

from aabb import AABB
from primitive.primitive import Primitive
from utils.typecheck import is_3d_array, is_float
from utils.utils import format_xyz


class Sphere(Primitive):
    """Class to create a sphere to render."""

    # ---------------------------------- Fields ---------------------------------- #
    _center: NDArray[np.float32]
    _radius: float

    # ------------------------------ Dunder methods ------------------------------ #
    def __init__(
        self,
        center: list[float] | NDArray[np.float32] = None,
        radius: float | int = 1.0,
        material: int = 0,
    ) -> None:
        """
        Initialization method.

        Args:
            center (list[float] | NDArray[np.float32]): Center coordinates of the sphere. Defaults to [0.0, 0.0, 0.0]
            radius (float | int): Radius of the sphere. Defaults to 1.0
            material (int, optional): Material index. Defaults to 0.

        """  # noqa: E501
        if center is None:
            center = [0.0, 0.0, 0.0]

        self._dtype = np.dtype(
            {
                "names": ["center", "radius", "radius2", "material", "_pad"],
                "formats": [
                    (np.dtype("<f4"), (3,)),
                    np.dtype("<f4"),
                    np.dtype("<f4"),
                    np.dtype("<u4"),
                    (np.dtype("<f4"), (2,)),
                ],
                "offsets": [0, 12, 16, 20, 24],
                "itemsize": 32,
            }
        )

        # create the private fields
        self._center = None
        self._radius = None

        super().__init__(material=material)

        self.center = center
        self.radius = radius

    def __str__(self) -> str:
        """
        Dunder method for converting the object into a printable string.

        Returns:
            str: Formatted string representing the object

        """
        output: str = "Sphere:\n"

        output += f"\tCenter: {format_xyz(self.center)}\n"
        output += f"\tRadius: {self.radius}\n"
        output += f"\tAABB: {self.aabb}"

        return output

    def __repr__(self) -> str:
        """
        Dunder method for converting the object into a printable string.

        Returns:
            str: Formatted string representing the object

        """
        output: str = "Sphere:\n"

        output += f"\tCenter: {format_xyz(self.center)}\n"
        output += f"\tRadius: {self.radius}\n"
        output += f"\tAABB: {self.aabb}"

        return output

    # ---------------------------- Setters and Getters --------------------------- #
    @property
    def center(self) -> NDArray[np.float32]:
        """
        Center coordinates of the sphere.

        Returns:
            NDArray[np.float32]: Array containing the X, Y, Z coordinates of the sphere

        """
        return self._center

    @center.setter
    def center(self, value: NDArray[np.float32] | list[float]) -> None:
        self._center = is_3d_array(value)

        self._calculate_aabb()
        self._calculate_ogl_ssbo_array()

    @property
    def radius(self) -> float:
        """
        Radius of the sphere.

        Returns:
            float: The radius of the sphere

        """
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        self._radius = is_float(value)

        self._calculate_aabb()
        self._calculate_ogl_ssbo_array()

    # ------------------------------ Private methods ----------------------------- #
    def _calculate_aabb(self) -> None:
        """Private method for calculating the AABB of the Sphere."""
        # if the properties of the sphere aren't initialized, skip the execution
        if self._center is None or self._radius is None:
            return

        # calculate the minimum and maximum coordinates reached by the sphere
        min_coords: NDArray[np.float32] = self._center - self._radius
        max_coords: NDArray[np.float32] = self._center + self._radius
        # store the extrenes in the AABB
        self._aabb = AABB(min_coords=min_coords, max_coords=max_coords)

    def _calculate_ogl_ssbo_array(self) -> None:
        """Private method for calculating the data array for OpenGL SSBO."""
        # if the properties of the sphere aren't initialized, skip the execution
        if self._center is None or self._radius is None:
            return

        # reset the data array
        self._ogl_ssbo_data = np.zeros(1, dtype=self._dtype)

        # add the values to the array with the specified structure
        self._ogl_ssbo_data[0]["center"] = [self._center[0], self._center[1], self._center[2]]
        self._ogl_ssbo_data[0]["radius"] = self._radius
        self._ogl_ssbo_data[0]["radius2"] = self._radius * self._radius
        self._ogl_ssbo_data[0]["material"] = self._material
        self._ogl_ssbo_data[0]["_pad"] = [0.0, 0.0]  # padding float
