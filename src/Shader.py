from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader


class Shader:
    def __init__(self, vertex_path = "", fragment_path = ""):
        self.program = None
        self.vertex_path = None
        self.fragment_path = None
        self.vertex_src = None
        self.fragment_src = None

        if len(vertex_path) != 0 and len(fragment_path) != 0:
            self.compile_program(vertex_path, fragment_path)

    def compile_program(self, vertex_path, fragment_path):
        self.vertex_path = vertex_path
        self.fragment_path = fragment_path

        with open(vertex_path) as f:
            self.vertex_src = f.readlines()
        with open(fragment_path) as f:
            self.fragment_src = f.readlines()

        self.program = compileProgram(
            compileShader(self.vertex_src, GL_VERTEX_SHADER),
            compileShader(self.fragment_src, GL_FRAGMENT_SHADER),
        )
