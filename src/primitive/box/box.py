"""box module."""

import numpy as np
from numpy.typing import NDArray

from aabb import AABB
from primitive.primitive import Primitive
from utils.typecheck import is_3d_array


class Box(Primitive):
    """Box class."""

    # ---------------------------------- Fields ---------------------------------- #
    _p0: NDArray[np.float32]
    _p1: NDArray[np.float32]

    # ------------------------------ Dundet methods ------------------------------ #
    def __init__(
        self,
        p0: NDArray[np.float32] | list[float] = None,
        p1: NDArray[np.float32] | list[float] = None,
        material: int | float = 0,
    ) -> None:
        """
        Initialization method.

        Args:
            p0 (NDArray[np.float32] | list[float], optional): First extreme vertex of the box. Defaults to [0.0, 0.0, 0.0].
            p1 (NDArray[np.float32] | list[float], optional): Second extreme vertex of the box. Defaults to [1.0, 1.0, 1.0].
            material (int | float, optional): _description_. Defaults to 0.

        """  # noqa: E501
        if p0 is None:
            p0 = [0.0, 0.0, 0.0]
        if p1 is None:
            p1 = [1.0, 1.0, 1.0]

        self._dtype = np.dtype(
            [
                ("p0", np.float32, 3),
                ("material", np.uint32),
                ("p1", np.float32, 3),
                ("_pad", np.float32),
            ],
            align=True,
        )

        self._p0 = None
        self._p1 = None

        super().__init__(material=material)

        self.p0 = p0
        self.p1 = p1

    # ---------------------------- Setters and getters --------------------------- #
    @property
    def p0(self) -> NDArray[np.float32]:
        """
        First point of the box.

        Returns:
            NDArray[np.float32]: Array containing the X, Y, Z coordinates of the point.

        """
        return self._p0

    @p0.setter
    def p0(self, value: NDArray[np.float32] | list[float]) -> None:
        self._p0 = is_3d_array(value)

        self._calculate_aabb()
        self._calculate_ogl_ssbo_array()

    @property
    def p1(self) -> NDArray[np.float32]:
        """
        Second point of the box.

        Returns:
            NDArray[np.float32]: Array containing the X, Y, Z coordinates of the point.

        """
        return self._p1

    @p1.setter
    def p1(self, value: NDArray[np.float32] | list[float]) -> None:
        self._p1 = is_3d_array(value)

        self._calculate_aabb()
        self._calculate_ogl_ssbo_array()

    # ------------------------------ Private methods ----------------------------- #
    def _calculate_aabb(self) -> None:
        """Private method for calculating the AABB of the Box."""
        # if the properties of the box aren't initialized, skip the execution
        if self._p0 is None or self._p1 is None:
            return

        # calculate the minimum and maximum coordinates reached by the box
        min_coords: NDArray[np.float32] = np.minimum(self._p0, self._p1)
        max_coords: NDArray[np.float32] = np.maximum(self._p0, self._p1)
        # store the extremes in the AABB
        self._aabb = AABB(min_coords=min_coords, max_coords=max_coords)

    def _calculate_ogl_ssbo_array(self) -> None:
        """Private method for calculating the data array for OpenGL SSBO."""
        # if the properties of the box aren't initialized, skip the execution
        if self._p0 is None or self._p1 is None:
            return

        # reset the data array
        self._ogl_ssbo_data = np.zeros(1, dtype=self._dtype)

        # add the values to the array with the specified structure
        self._ogl_ssbo_data[0]["p0"] = self._p0
        self._ogl_ssbo_data[0]["p1"] = self._p1
        self._ogl_ssbo_data[0]["material"] = self._material
        self._ogl_ssbo_data[0]["_pad"] = 0.0  # padding float
