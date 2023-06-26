from PIL import Image
import numpy as np
import colorsys
from OpenGL.GL import *
from ctypes import *

from Shader import Shader

import time
import win_precise_time as wpt

import pywavefront

from pyrr import Matrix44

from model.Model import Model
from camera.Camera import Camera

class Renderer:
    
    __instance = None
    
    @staticmethod
    def getInstance():
        if Renderer.__instance == None:
            Renderer()
        return Renderer.__instance
    
    def __init__(self):
        if Renderer.__instance != None:
            raise Exception("Window already exists!")
        
        Renderer.__instance = self

        glClearColor(0.1, 0.1, 0.1, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # glEnable(GL_DEPTH_TEST)

        self.shader = Shader("../shaders/vertex.glsl", "../shaders/fragment.glsl")

        vertices = [
            -1, -1, 0.0,
             1, -1, 0.0,
             1,  1, 0.0,
            -1, -1, 0.0,
            -1,  1, 0.0,
             1,  1, 0.0
        ]

        vertices = (GLfloat * len(vertices))(*vertices)

        self.vbo = None
        self.vbo = glGenBuffers(1, self.vbo)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW)


        self.vao = None
        self.vao = glGenVertexArrays(1, self.vao)

        glBindVertexArray(self.vao)

        glEnableVertexAttribArray(0)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), c_void_p(0))


        self.vertices = glGenBuffers(1)
        self.model_mats = glGenBuffers(1)
        self.indices = glGenBuffers(1)
        self.colors = glGenBuffers(1)
        self.normals = glGenBuffers(1)

        self.camera = Camera.getInstance()

        self.render_time = 0

        self.light_index = 2

        self.light_model = Matrix44.identity()

    
    def render(self):
        start = wpt.time()

        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.shader.program)
        glUniformMatrix4fv(glGetUniformLocation(self.shader.program, "inInverseViewProjection"), 1, GL_FALSE, self.camera.get_inv_view_proj_matrix())
        glUniform3f(glGetUniformLocation(self.shader.program, "inEye"), self.camera.position[0], self.camera.position[1], self.camera.position[2])
        glUniform1f(glGetUniformLocation(self.shader.program, "inLightIndex"), self.light_index)
        glUniformMatrix4fv(glGetUniformLocation(self.shader.program, "inLightModel"), 1, GL_FALSE, self.light_model)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        end = wpt.time()
        self.render_time = end - start

        
    def update_vertices(self, vertices):
        data = (GLfloat * len(vertices))(*vertices)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.vertices)
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(data), data, GL_DYNAMIC_COPY)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.vertices)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def update_indices(self, indices):
        data = (GLfloat * len(indices))(*indices)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.indices)
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(data), data, GL_DYNAMIC_COPY)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self.indices)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def update_model_mats(self, model_mats):
        data = (GLfloat * len(model_mats))(*model_mats)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.model_mats)
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(data), data, GL_DYNAMIC_COPY)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self.model_mats)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        
    def update_colors(self, colors):
        data = (GLfloat * len(colors))(*colors)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.colors)
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(data), data, GL_DYNAMIC_COPY)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, self.colors)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def update_normals(self, normals):
        data = (GLfloat * len(normals))(*normals)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.normals)
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(data), data, GL_DYNAMIC_COPY)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, self.normals)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)


def normalize(vector):
    return vector / np.linalg.norm(vector)
