"""Utils module."""

import numpy as np
from numpy.typing import NDArray

color_codes: dict[str, str] = {
    "reset": "\033[0m",
    "red": "\033[31m",
    "green": "\033[32m",
    "blue": "\033[34m",
}


def format_xyz(data: list | NDArray) -> str:
    """
    Function to format a 3 element list into a color coded x, y, z string.

    Args:
        data (list | NDArray): Data to convert into a string

    Returns:
        str: Color coded X, Y, Z string

    """
    output: str = ""

    output += f"[{color_codes['red']}{data[0]}{color_codes['reset']}, "
    output += f"{color_codes['green']}{data[1]}{color_codes['reset']}, "
    output += f"{color_codes['blue']}{data[2]}{color_codes['reset']}]"

    return output


def normalize(vector: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Normalize a 3D Vector.

    Args:
        vector (NDArray[np.float32]): 3D Vector

    Returns:
        NDArray[np.float32]: Normalized 3D Vector.

    """
    return vector / np.linalg.norm(vector)


def pack_data_30_2(first_value: int, second_value: int) -> int:
    """
    Packs a 30-bit prim_idx and a 2-bit prim_type into a single 32-bit integer.

    The prim_type is stored in the 2 most significant bits.

    Args:
        first_value (int): Value to pack on the first 30 bits
        second_value (int): Value to pack on the last 2 bits

    Raises:
        ValueError: In case the provided values don't fit in the specified bits

    Returns:
        int: Packed value

    """
    # Ensure prim_idx fits in 30 bits and prim_type in 2 bits
    if not (0 <= first_value < (1 << 30)):
        first_value = 1073741823
    if not (0 <= second_value < (1 << 2)):
        second_value = 3

    # Shift the type by 30 bits and combine with the index using bitwise OR
    return (second_value << 30) | first_value


def pack_data_28_2_2(first_value: int, second_value: int, third_value: int) -> int:
    """
    Packs three unsigned integers into a single 32-bit integer.

      - first_value  (28 bits, least significant)
      - second_value (2 bits, middle)
      - third_value  (2 bits, most significant)

    Args:
        first_value (int): Value for the lower 28 bits
        second_value (int): Value for the middle 2 bits
        third_value (int): Value for the upper 2 bits

    Raises:
        ValueError: If any value doesn't fit in the specified bit width

    Returns:
        int: Packed 32-bit integer

    """
    if not (0 <= first_value < (1 << 28)):
        first_value = 268435455
    if not (0 <= second_value < (1 << 2)):
        second_value = 3
    if not (0 <= third_value < (1 << 2)):
        third_value = 3

    return (third_value << 30) | (second_value << 28) | first_value


def pack_4_floats_to_uint32(values: NDArray[np.float32]) -> NDArray[np.uint32]:
    clipped_floats: NDArray[np.float32] = np.clip(values, 0.0, 1.0)
    quantized_uint8s: NDArray[np.uint8] = np.round(clipped_floats * 255).astype(np.uint8)
    packed_uint32: NDArray[np.uint32] = quantized_uint8s.view(np.uint32)[0]

    return packed_uint32


def pack_r11g11b10(values: NDArray[np.float32]) -> NDArray[np.uint32]:
    """
    Pack RGB floats in [0,1] into a 32-bit integer with R10G11B10 format.
    R = 10 bits, G = 11 bits, B = 10 bits.
    """
    clipped = np.clip(values, 0.0, 1.0)

    # Quantize each channel
    r = np.round(clipped[..., 0] * ((1 << 11) - 1)).astype(np.uint32)  # 11 bits
    g = np.round(clipped[..., 1] * ((1 << 11) - 1)).astype(np.uint32)  # 11 bits
    b = np.round(clipped[..., 2] * ((1 << 10) - 1)).astype(np.uint32)  # 10 bits

    # Pack into uint32
    packed = (r << (11 + 10)) | (g << 10) | b
    return packed.astype(np.uint32)


# Print iterations progress
def print_progress_bar(  # noqa: PLR0913
    iteration: int | float,
    total: int | float,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 100,
    fill: str = "█",
    print_end: str = "\r",
) -> None:
    r"""
    Call in a loop to create terminal progress bar.

    Args:
        iteration (int | float): current iteration
        total (int | float): total iterations
        prefix (str): prefix string. Defaults to ''
        suffix (str): suffix string. Defaults to ''
        decimals (int): positive number of decimals in percent complete. Defaults to 1
        length (int): character length of bar. Defaults to 100
        fill (str): bar fill character. Defaults to █
        print_end (str): end character (e.g. "\r", "\r\n"). Defaults to \r

    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


def quad_to_triangles(center, width, height, normal):
    center = np.array(center, dtype=float)
    normal = np.array(normal, dtype=float)
    normal /= np.linalg.norm(normal)

    # Pick an arbitrary reference vector not parallel to normal
    reference = np.array([1.0, 0.0, 0.0])
    if np.allclose(normal, reference) or np.allclose(normal, -reference):
        reference = np.array([0.0, 1.0, 0.0])

    tangent = np.cross(normal, reference)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)

    # Scale by half width and height
    half_width = width / 2.0
    half_height = height / 2.0
    tangent *= half_width
    bitangent *= half_height

    # Quad corners
    p0 = center - tangent - bitangent
    p1 = center + tangent - bitangent
    p2 = center + tangent + bitangent
    p3 = center - tangent + bitangent

    # Two triangles (counter-clockwise)
    tri1 = [p0.tolist(), p1.tolist(), p2.tolist()]
    tri2 = [p0.tolist(), p2.tolist(), p3.tolist()]

    return [tri1, tri2]
