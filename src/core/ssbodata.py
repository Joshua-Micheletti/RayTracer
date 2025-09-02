"""ssbo_data module."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class SSBOData(ABC):
    """Abstract class that provides an OpenGL SSBO data interface."""

    _dtype: np.dtype
    _ogl_ssbo_data: NDArray

    def __init__(self) -> None:
        """Initialization method."""
        self._ogl_ssbo_data = np.zeros(1, self._dtype)

    @property
    def dtype(self) -> np.dtype:
        """
        Numpy dtype for the formatted data of the sphere.

        Returns:
            np.dtype: Type of the sphere data

        """
        return self._dtype

    @property
    def ogl_ssbo_data(self) -> NDArray:
        """
        Data representing the primitive formatted to STD430 OpenGL SSBO standard layout.

        Returns:
            NDarray[self._dtype]: Formatted array

        """
        return self._ogl_ssbo_data

    @abstractmethod
    def _calculate_ogl_ssbo_array(self) -> None:
        pass
