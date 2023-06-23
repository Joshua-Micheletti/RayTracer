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

        self.ssbo = None;
        self.ssbo = glGenBuffers(1);
        

        self.model = Model("../models/tree.obj")
        self.model.scale(0.05, 0.05, 0.05)
        self.model.move(0, -1, 0)

        shader_data = (GLfloat * len(self.model.vertices))(*self.model.vertices)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(shader_data), shader_data, GL_DYNAMIC_COPY)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.ssbo)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        self.camera = Camera.getInstance()

        self.render_time = 0

    
    def render(self):
        start = wpt.time()

        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.shader.program)
        glUniformMatrix4fv(glGetUniformLocation(self.shader.program, "inModel"), 1, GL_FALSE, self.model.model_matrix)
        # glUniformMatrix4fv(glGetUniformLocation(self.shader.program, "inView"), 1, GL_FALSE, self.camera.get_view_matrix())
        # glUniformMatrix4fv(glGetUniformLocation(self.shader.program, "inProjection"), 1, GL_FALSE, self.camera.projection_matrix)
        glUniformMatrix4fv(glGetUniformLocation(self.shader.program, "inInverseViewProjection"), 1, GL_FALSE, self.camera.get_inv_view_proj_matrix())
        glUniform3f(glGetUniformLocation(self.shader.program, "inEye"), self.camera.position[0], self.camera.position[1], self.camera.position[2])
        # glUniform3f(glGetUniformLocation(self.shader.program, "inUp"), self.camera.up[0], self.camera.up[1], self.camera.up[2])
        # glUniform3f(glGetUniformLocation(self.shader.program, "inRight"), self.camera.right[0], self.camera.right[1], self.camera.right[2])
        # glUniform3f(glGetUniformLocation(self.shader.program, "inFront"), self.camera.front[0], self.camera.front[1], self.camera.front[2])
        glUniform3f(glGetUniformLocation(self.shader.program, "inLight"), 2.0, 2.0, 1.0)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        end = wpt.time()
        self.render_time = end - start

        

        
        
        
        
def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


def normalize(vector):
    return vector / np.linalg.norm(vector)


def inside_outside_test(v0, v1, v2, p, n):
    edge0 = v1 - v0
    edge1 = v2 - v1
    edge2 = v0 - v2

    c0 = p - v0
    c1 = p - v1
    c2 = p - v2

    # print(c0)

    if np.dot(n, np.cross(edge0, c0)) > 0 and np.dot(n, np.cross(edge1, c1)) > 0 and np.dot(n, np.cross(edge2, c2)) > 0:
        return True

    return False


def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None


def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance


def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis
