from PIL import Image
import numpy as np
import colorsys
from OpenGL.GL import *
from ctypes import *

from Shader import Shader

import time
import win_precise_time as wpt

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
        glDisable(GL_DEPTH_TEST)

        self.shader = Shader("../shaders/vertex.glsl", "../shaders/fragment.glsl")

        vertices = [
            -0.5, -0.5, 0.0,
             0.5, -0.5, 0.0,
             0.0,  0.5, 0.0
        ]

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
    
    
    def render(self):
        start = wpt.time()

        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.shader.program)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        end = wpt.time()
        print(f"Pixel Shader: {(end - start) * 1000}")
        
        
        start = wpt.time()
        '''
        width = 200
        height = 160
        
        ratio = float(width) / height
        
        img = Image.new(mode = "RGB", size = (width, height), color = (77, 77, 77))
        
        models = [
            {'vertices': np.array([[-0.5, -0.5, 0],
                                   [ 0.5, -0.5, 0],
                                   [   0,  0.5, 0]])}
        ]

        for model in models:
            edge_01 = model["vertices"][1] - model["vertices"][0]
            edge_02 = model["vertices"][2] - model["vertices"][0]

            normal = np.cross(edge_01, edge_02)

            model["normal"] = normalize(normal)
        

        camera = np.array([0, 0, -1])

        for model in models:
            for y in range(height):
                for x in range(width):
                    u = map_range(x, 0, width,  0, 1)
                    v = map_range(y, 0, height, 0, 1)

                    u -= 0.5
                    v -= 0.5

                    u *= 2
                    v *= 2

                    v /= ratio

                    pixel = np.array([u, v, 0])

                    origin = camera
                    direction = normalize(pixel - camera)
                    #direction = normalize(np.array([1, 1, 0]) - camera)

                    normal = model["normal"]
                    vertices = model["vertices"]


                    distance = -np.dot(normal, vertices[0])

                    parallelism = np.dot(normal, direction)

                    if parallelism == 0 or np.isnan(parallelism):
                        print("division by 0")
                        continue


                    t = -(np.dot(normal, origin) + distance) / np.dot(normal, direction)

                    #print(f"t: {t}")

                    if t <= 0:
                        continue

                    p_hit = origin + t * direction

                    #print(p_hit)

                    if inside_outside_test(vertices[0], vertices[1], vertices[2], p_hit, normal):
                        #print("hit")
                        r = 30
                        g = 125
                        b = 255
                    
                    else:
                        r = 77
                        g = 77
                        b = 77

                    #r = int(map_range(distance, 0, 1, 0, 255))
                    #g = int(map_range(distance, 0, 1, 0, 255))
                    #b = int(map_range(distance, 0, 1, 0, 255))

                    #r = int(map_range(clamp((p_hit - vertices[0])[0], 0, 1), 0, 1, 0, 255))
                    #g = int(map_range(clamp((p_hit - vertices[0])[1], 0, 1), 0, 1, 0, 255));
                    #b = int(map_range(clamp((p_hit - vertices[0])[2], 0, 1), 0, 1, 0, 255));

            # intersectionPoint = origin + t * direction


                    #r = abs(int(map_range(u, 0, 1, 0, 255)))
                    #g = abs(int(map_range(v, 0, 1, 0, 255)))
                    #b = 0

                    img.putpixel((x, (height - 1) - y), (r, g, b))


        img.save("render.png")
        end = wpt.time()

        print(f"CPU: {(end - start) * 1000}")
        '''
        
        
        
        
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
