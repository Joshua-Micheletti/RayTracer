from PIL import Image
import numpy as np
import colorsys
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from ctypes import *

from Shader import Shader

import time
import win_precise_time as wpt

import pywavefront

from pyrr import Matrix44

from model.Model import Model
from camera.Camera import Camera
import random

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

        # self.shader = Shader("../shaders/vertex.glsl", "../shaders/fragment.glsl")

        vertices = [
            -1.0,  1.0, 0.0, 0.0, 1.0,
			-1.0, -1.0, 0.0, 0.0, 0.0,
			 1.0,  1.0, 0.0, 1.0, 1.0,
			 1.0, -1.0, 0.0, 1.0, 0.0
        ]

        vertices = (GLfloat * len(vertices))(*vertices)

        self.vbo = None
        self.vao = None

        self.vao = glGenVertexArrays(1, self.vao)
        self.vbo = glGenBuffers(1, self.vbo)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), c_void_p(3 * sizeof(GLfloat)))

        self.vertices = glGenBuffers(1)
        self.model_mats = glGenBuffers(1)
        self.indices = glGenBuffers(1)
        self.colors = glGenBuffers(1)
        self.normals = glGenBuffers(1)
        self.spheres = glGenBuffers(1)
        self.sphere_colors = glGenBuffers(1)
        self.planes = glGenBuffers(1)
        self.plane_colors = glGenBuffers(1)
        self.boxes = glGenBuffers(1)
        self.boxes_colors = glGenBuffers(1)
        self.swap = glGenBuffers(1)
        self.bounding_boxes = glGenBuffers(1)


        self.camera = Camera.getInstance()

        self.render_time = 0

        self.light_index = 2

        self.light_model = Matrix44.identity()

        compute_shader_source = ""

        self.compute_shader = glCreateShader(GL_COMPUTE_SHADER)

        text = open("../shaders/compute.glsl", 'r')
        compute_shader_source = text.read()

        glShaderSource(self.compute_shader, compute_shader_source)
        glCompileShader(self.compute_shader)

        status = None
        glGetShaderiv(self.compute_shader, GL_COMPILE_STATUS, status)
        if status == GL_FALSE:
            # Note that getting the error log is much simpler in Python than in C/C++
            # and does not require explicit handling of the string buffer
            strInfoLog = glGetShaderInforLog(self.compute_shader)
            strShaderType = "compute"
            
            print("Compilation failure for " + strShaderType + " shader:\n" + strInfoLog)

        self.program_id = glCreateProgram()
        glAttachShader(self.program_id, self.compute_shader)
        glLinkProgram(self.program_id)
        print(glGetProgramInfoLog(self.program_id))

        self.texture = 0
        self.texture = glGenTextures(1, self.texture)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, 384, 216)
        glBindImageTexture(0, self.texture, 1, GL_TRUE, 0, GL_READ_WRITE, GL_RGBA32F)

        self.last_frame = 0
        self.last_frame = glGenTextures(1, self.last_frame)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.last_frame)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, 384, 216)
        glBindImageTexture(0, self.last_frame, 1, GL_TRUE, 0, GL_READ_WRITE, GL_RGBA32F)

        self.screen_shader = Shader("../shaders/screen_vertex.glsl", "../shaders/screen_fragment.glsl")

        self.rendered_frames = 0


    def render(self):
        start = wpt.time()

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glBindImageTexture(0, self.texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F)

        glUseProgram(self.program_id)

        glUniformMatrix4fv(glGetUniformLocation(self.program_id, "inverse_view_projection"), 1, GL_FALSE, self.camera.get_inv_view_proj_matrix())
        glUniform3f(glGetUniformLocation(self.program_id, "eye"), self.camera.position[0], self.camera.position[1], self.camera.position[2])
        glUniform1f(glGetUniformLocation(self.program_id, "light_index"), self.light_index)
        glUniformMatrix4fv(glGetUniformLocation(self.program_id, "light_model"), 1, GL_FALSE, self.light_model)
        glUniform1f(glGetUniformLocation(self.program_id, "random_1"), random.random())
        glUniform1f(glGetUniformLocation(self.program_id, "random_2"), random.random())
        glUniform1f(glGetUniformLocation(self.program_id, "random_3"), random.random())
        # print(time.time() / 100000000)
        print(random.random())

        glDispatchCompute(int(384 / 8), int(216 / 4), 1)
        glMemoryBarrier(GL_ALL_BARRIER_BITS)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.screen_shader.program)
        glUniform1i(glGetUniformLocation(self.screen_shader.program, "tex"), 0)
        glUniform1i(glGetUniformLocation(self.screen_shader.program, "old_tex"), 1)
        glUniform1f(glGetUniformLocation(self.screen_shader.program, "frames"), self.rendered_frames)
        
        glBindVertexArray(self.vao)

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)

        self.rendered_frames += 1

        glBindTexture(GL_TEXTURE_2D, self.last_frame)
        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, 384, 216)

        # glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0,)
        # glCopyImageSubData(self.texture, GL_TEXTURE_2D, 0, 0, 0, 0, self.last_frame, GL_TEXTURE_2D, 0, 0, 0, 0, 384, 216, 1)

    

        end = wpt.time()
        self.render_time = end - start

        
    def update_vertices(self, vertices):
        data = (GLfloat * len(vertices))(*vertices)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.vertices)
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.vertices)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def update_indices(self, indices):
        data = (GLfloat * len(indices))(*indices)

        glBindBuffer(GL_UNIFORM_BUFFER, self.indices)
        glBufferData(GL_UNIFORM_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 3, self.indices)
        glBindBuffer(GL_UNIFORM_BUFFER, 0)

    def update_model_mats(self, model_mats):
        data = (GLfloat * len(model_mats))(*model_mats)

        glBindBuffer(GL_UNIFORM_BUFFER, self.model_mats)
        glBufferData(GL_UNIFORM_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 2, self.model_mats)
        glBindBuffer(GL_UNIFORM_BUFFER, 0)
        
    def update_colors(self, colors):
        data = (GLfloat * len(colors))(*colors)

        glBindBuffer(GL_UNIFORM_BUFFER, self.colors)
        glBufferData(GL_UNIFORM_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 4, self.colors)
        glBindBuffer(GL_UNIFORM_BUFFER, 0)

    def update_normals(self, normals):
        data = (GLfloat * len(normals))(*normals)

        glBindBuffer(GL_UNIFORM_BUFFER, self.normals)
        glBufferData(GL_UNIFORM_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 5, self.normals)
        glBindBuffer(GL_UNIFORM_BUFFER, 0)

    def update_spheres(self, spheres):
        data = (GLfloat * len(spheres))(*spheres)

        glBindBuffer(GL_UNIFORM_BUFFER, self.spheres)
        glBufferData(GL_UNIFORM_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 6, self.spheres)
        glBindBuffer(GL_UNIFORM_BUFFER, 0)

    def update_sphere_colors(self, colors):
        data = (GLfloat * len(colors))(*colors)

        glBindBuffer(GL_UNIFORM_BUFFER, self.sphere_colors)
        glBufferData(GL_UNIFORM_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 7, self.sphere_colors)
        glBindBuffer(GL_UNIFORM_BUFFER, 0)

    def update_planes(self, planes):
        data = (GLfloat * len(planes))(*planes)

        glBindBuffer(GL_UNIFORM_BUFFER, self.planes)
        glBufferData(GL_UNIFORM_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 8, self.planes)
        glBindBuffer(GL_UNIFORM_BUFFER, 0)

    def update_plane_colors(self, colors):
        data = (GLfloat * len(colors))(*colors)

        glBindBuffer(GL_UNIFORM_BUFFER, self.plane_colors)
        glBufferData(GL_UNIFORM_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 9, self.plane_colors)
        glBindBuffer(GL_UNIFORM_BUFFER, 0)

    def update_boxes(self, boxes):
        data = (GLfloat * len(boxes))(*boxes)

        glBindBuffer(GL_UNIFORM_BUFFER, self.boxes)
        glBufferData(GL_UNIFORM_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 10, self.boxes)
        glBindBuffer(GL_UNIFORM_BUFFER, 0)

    def update_boxes_colors(self, colors):
        data = (GLfloat * len(colors))(*colors)

        glBindBuffer(GL_UNIFORM_BUFFER, self.boxes_colors)
        glBufferData(GL_UNIFORM_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 11, self.boxes_colors)
        glBindBuffer(GL_UNIFORM_BUFFER, 0)

    def update_bounding_boxes(self, bounding_boxes):
        # print(bounding_boxes)
        data = (GLfloat * len(bounding_boxes))(*bounding_boxes)

        glBindBuffer(GL_UNIFORM_BUFFER, self.bounding_boxes)
        glBufferData(GL_UNIFORM_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
        glBindBufferBase(GL_UNIFORM_BUFFER, 12, self.bounding_boxes)
        glBindBuffer(GL_UNIFORM_BUFFER, 0)


    def reset_accumulation(self):
        glCopyImageSubData(self.texture, GL_TEXTURE_2D, 0, 0, 0, 0, self.last_frame, GL_TEXTURE_2D, 0, 0, 0, 0, 384, 216, 1)
        self.rendered_frames = 0




def normalize(vector):
    return vector / np.linalg.norm(vector)
