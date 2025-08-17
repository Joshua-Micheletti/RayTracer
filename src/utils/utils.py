"""Utils module."""

from numpy.typing import NDArray

color_codes: dict[str, str] = {
    "reset": "\033[0m",
    "red": "\033[31m",
    "green": "\033[32m",
    "blue": "\033[34m",
}

def format_xyz(data: list | NDArray) -> str:
    """Function to format a 3 element list into a color coded x, y, z string.

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
