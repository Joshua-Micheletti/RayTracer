"""Module for typechecking variables and making them safe."""

import numpy as np
from numpy.typing import NDArray


def is_float32_array(value: NDArray[np.float32] | list[float]) -> NDArray[np.float32]:
    """
    Function to typecheck a float array.

    Args:
        value (NDArray[np.float32] | list[float]): Input

    Raises:
        TypeError: In case the value is not an array of floats

    Returns:
        NDArray[np.float32]: Typesafe array

    """
    if isinstance(value, list):
        return np.array(value, dtype=np.float32)

    if isinstance(value, np.ndarray) and value.dtype == np.float32:
        return value

    message: str = "Value must be a list of float or an numpy array of float 32"
    raise TypeError(message)


def is_3d_array(value: NDArray[np.float32] | list[float]) -> NDArray[np.float32]:
    """
    Function to typecheck a 3D array.

    Args:
        value (NDArray[np.float32] | list[float]): Input

    Raises:
        TypeError: In case the value is not an array or its length is not 3

    Returns:
        NDArray[np.float32]: Typesafe array

    """
    safe_value: NDArray[np.float32] = is_float32_array(value)

    if safe_value.shape == (3,):
        return safe_value

    message: str = "Value must contain 3 values, one for each of the X, Y, Z components"
    raise TypeError(message)

def is_4d_array(value: NDArray[np.float32] | list[float]) -> NDArray[np.float32]:
    """
    Function to typecheck a 4D array.

    Args:
        value (NDArray[np.float32] | list[float]): Input

    Raises:
        TypeError: In case the value is not an array or its length is not 3

    Returns:
        NDArray[np.float32]: Typesafe array

    """
    safe_value: NDArray[np.float32] = is_float32_array(value)

    if safe_value.shape == (4,):
        return safe_value

    message: str = "Value must contain 4 values, one for each of the X, Y, Z, W components"
    raise TypeError(message)


def is_float(value: float | int) -> float:
    """
    Function to typecheck a float variable.

    Args:
        value (float | int): Input

    Raises:
        TypeError: In case the value isn't a float or it can't be converted to a float

    Returns:
        float: Typesafe value

    """
    if isinstance(value, int):
        return float(value)

    if isinstance(value, float):
        return value

    message: str = "Value must be float"
    raise TypeError(message)


def is_int(value: int | float) -> int:
    """
    Function to typecheck an int variable.

    Args:
        value (int | float): Input

    Raises:
        TypeError: In case the value isn't an int or it can't be converted to an int

    Returns:
        float: Typesafe value

    """
    if isinstance(value, float):
        return int(value)

    if isinstance(value, int):
        return value

    message: str = "Value must be int"
    raise TypeError(message)


def is_positive_int(value: int | float) -> int:
    """
    Function to typecheck a positive int variable.

    Args:
        value (int | float): Input

    Raises:
        TypeError: In case the value isn't a positive int

    Returns:
        int: Typesafe value

    """
    safe_value = is_int(value)

    if safe_value >= 0:
        return safe_value

    message: str = "Value must be a positive int"
    raise TypeError(message)
