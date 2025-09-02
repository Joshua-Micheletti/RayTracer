"""shader module."""

from pathlib import Path

from OpenGL import GL
from OpenGL.GL.shaders import compileProgram, compileShader


class Shader:
    """Shader class."""

    def __init__(self, vertex_path: str = "", fragment_path: str = "") -> None:
        """
        Initialization method.

        Args:
            vertex_path (str, optional): Path to the vertex shader. Defaults to "".
            fragment_path (str, optional): Path to the fragment shader. Defaults to "".

        """
        self.program = None
        self.vertex_path = None
        self.fragment_path = None
        self.vertex_src = None
        self.fragment_src = None

        if len(vertex_path) != 0 and len(fragment_path) != 0:
            self.compile_program(vertex_path, fragment_path)

    def compile_program(self, vertex_path: str, fragment_path: str) -> None:
        """
        Method for compiling a vertex and fragment shader.

        Args:
            vertex_path (str): Path to the vertex shader
            fragment_path (str): Path to the fragment shader

        """
        self.vertex_path = vertex_path
        self.fragment_path = fragment_path

        with Path(vertex_path).open("r") as f:
            self.vertex_src = f.readlines()
        with Path(fragment_path).open("r") as f:
            self.fragment_src = f.readlines()

        self.program = compileProgram(
            compileShader(self.vertex_src, GL.GL_VERTEX_SHADER),
            compileShader(self.fragment_src, GL.GL_FRAGMENT_SHADER),
        )
