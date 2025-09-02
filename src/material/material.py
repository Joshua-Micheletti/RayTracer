"""material module."""

import numpy as np
from numpy.typing import NDArray

from core import SSBOData
from utils.typecheck import is_3d_array, is_4d_array, is_float
from utils.utils import format_xyz, pack_4_floats_to_uint32, pack_r11g11b10


class Material(SSBOData):
    """Material class."""

    # ---------------------------------- Fields ---------------------------------- #
    _color: NDArray[np.float32]
    _emission: NDArray[np.float32]
    _smoothness: float
    _ior: float
    _metallic: float
    _transmission: float

    # ------------------------------ Dunder methods ------------------------------ #
    def __init__(
        self,
        color: NDArray[np.float32] | list[float] = None,
        emission: NDArray[np.float32] | list[float] = None,
        smoothness: float = 0.0,
        ior: float = 1.0,
        metallic: float = 0.0,
        transmission: float = 0.0,
    ) -> None:
        """
        Initalization method.

        Args:
            color (NDArray[np.float32] | list[float]): Base color. Defaults to [1.0, 1.0, 1.0]
            emission (NDArray[np.float32] | list[float]): Emission color and strength. Defaults to [0.0, 0.0, 0.0, 0.0]
            smoothness (float): Material's tendency to reflect perfectly or scatter light. Defaults to 0.0

        """  # noqa: E501
        if emission is None:
            emission = [0.0, 0.0, 0.0, 0.0]
        if color is None:
            color = [1.0, 1.0, 1.0]

        u4 = np.dtype("<u4")  # little-endian uint32

        self._dtype = np.dtype(
            {
                "names": ["color", "emission", "properties"],
                "formats": [u4, u4, u4],
                "offsets": [0, 4, 8],
                "itemsize": 12,  # 3 x 4 bytes
            }
        )

        self._color = None
        self._emission = None
        self._smoothness = None
        self._metallic = None
        self._transmission = None
        self._ior = None

        super().__init__()

        self.color = color
        self.emission = emission
        self.smoothness = smoothness
        self.metallic = metallic
        self.transmission = transmission
        self.ior = ior

    def __str__(self) -> str:
        output = "Material:\n"
        output += f"Color: {format_xyz(self._color)}\n"
        output += (
            f"Emission: {format_xyz([self._emission[0], self._emission[1], self._emission[2]])}\n"
        )
        output += f"Smoothness: {self._smoothness}\n"
        output += f"Transmission: {self._transmission}\n"
        output += f"Metallic: {self._metallic}\n"
        output += f"Ior: {self._ior}\n"

        return output

    def __repr__(self) -> str:
        output = "Material:\n"
        output += f"Color: {format_xyz(self._color)}\n"
        output += (
            f"Emission: {format_xyz([self._emission[0], self._emission[1], self._emission[2]])}\n"
        )
        output += f"Smoothness: {self._smoothness}\n"
        output += f"Transmission: {self._transmission}\n"
        output += f"Metallic: {self._metallic}\n"
        output += f"Ior: {self._ior}\n"

        return output

    # ---------------------------- Setters and getters --------------------------- #
    @property
    def color(self) -> NDArray[np.float32]:
        """
        Color of the material.

        Returns:
            NDArray[np.float32]: 3D Array of RGB values

        """
        return self._color

    @color.setter
    def color(self, value: NDArray[np.float32] | list[float]) -> None:
        self._color = is_3d_array(value)
        self._calculate_ogl_ssbo_array()

    @property
    def emission(self) -> NDArray[np.float32]:
        """
        Emission of the material.

        Returns:
            NDArray[np.float32]: 4D Array of RGB-S(trength) values

        """
        return self._emission

    @emission.setter
    def emission(self, value: NDArray[np.float32] | list[float]) -> None:
        self._emission = is_4d_array(value)
        self._calculate_ogl_ssbo_array()

    @property
    def smoothness(self) -> float:
        """
        Smoothness of the material.

        Returns:
            float: Value between 0 and 1

        """
        return self._smoothness

    @smoothness.setter
    def smoothness(self, value: float | int) -> None:
        self._smoothness = is_float(value)
        self._calculate_ogl_ssbo_array()

    @property
    def metallic(self) -> float:
        """
        Metallicness of the material.

        Returns:
            float: Value between 0 and 1

        """
        return self._metallic

    @metallic.setter
    def metallic(self, value: float | int) -> None:
        self._metallic = is_float(value)
        self._calculate_ogl_ssbo_array()

    @property
    def transmission(self) -> float:
        """
        Opacity of the material.

        Returns:
            float: Value between 0 and 1

        """
        return self._transmission

    @transmission.setter
    def transmission(self, value: float | int) -> None:
        self._transmission = is_float(value)
        self._calculate_ogl_ssbo_array()

    @property
    def ior(self) -> float:
        """
        Index of Refraction of the material.

        Returns:
            float: Value between 0 and 1

        """
        return self._ior

    @transmission.setter
    def ior(self, value: float | int) -> None:
        self._ior = is_float(value)
        self._calculate_ogl_ssbo_array()

    # ------------------------------ Private methods ----------------------------- #
    def _calculate_ogl_ssbo_array(self) -> None:
        """Private method for calculating the data array for OpenGL SSBO."""
        # if the properties of the sphere aren't initialized, skip the execution
        if (
            self._color is None
            or self._emission is None
            or self._smoothness is None
            or self._metallic is None
            or self._transmission is None
            or self._ior is None
        ):
            return

        # reset the data array
        self._ogl_ssbo_data = np.zeros(1, dtype=self._dtype)

        clipped_floats: NDArray[np.float32] = np.clip(
            np.array([self._smoothness, self._metallic, self._transmission, self._ior]), 0.0, 1.0
        )
        quantized_uint8s: NDArray[np.uint8] = np.round(clipped_floats * 255).astype(np.uint8)
        quantized_uint8s[3] = np.uint8(np.round(np.clip(self._ior / 3.0, 0.0, 1.0) * 255))
        packed_uint32: NDArray[np.uint32] = quantized_uint8s.view(np.uint32)[0]

        # add the values to the array with the specified structure
        self._ogl_ssbo_data[0]["properties"] = packed_uint32

        self._ogl_ssbo_data[0]["color"] = pack_r11g11b10(self._color)

        clipped_floats: NDArray[np.float32] = np.clip(self._emission, 0.0, 1.0)
        quantized_uint8s: NDArray[np.uint8] = np.round(clipped_floats * 255).astype(np.uint8)
        quantized_uint8s[3] = self._emission[3].astype(np.uint8)
        packed_uint32: NDArray[np.uint32] = quantized_uint8s.view(np.uint32)[0]

        self._ogl_ssbo_data[0]["emission"] = packed_uint32
        # self._ogl_ssbo_data[0]["color_pad"] = 0
        # self._ogl_ssbo_data[0]["specular_pad"] = 0
        # self._ogl_ssbo_data[0]["pad"] = [0.0, 0.0]
