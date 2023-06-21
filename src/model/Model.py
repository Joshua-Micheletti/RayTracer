from pyrr import Matrix44
import numpy as np
import pyrr

import pywavefront


class Model:

    def __init__(self, filepath):
        self.x = 0
        self.y = 0
        self.z = 0

        self.scale_x = 1
        self.scale_y = 1
        self.scale_z = 1

        self.model_matrix = Matrix44.identity()
        self.translation_matrix = Matrix44.identity()
        self.scale_matrix = Matrix44.identity()

        self.vertices = []

        data = pywavefront.Wavefront(filepath, collect_faces = True)

        for mesh in data.mesh_list:
            for face in mesh.faces:
                self.vertices.append(data.vertices[face[0]][0])
                self.vertices.append(data.vertices[face[0]][1])
                self.vertices.append(data.vertices[face[0]][2])

                self.vertices.append(data.vertices[face[1]][0])
                self.vertices.append(data.vertices[face[1]][1])
                self.vertices.append(data.vertices[face[1]][2])

                self.vertices.append(data.vertices[face[2]][0])
                self.vertices.append(data.vertices[face[2]][1])
                self.vertices.append(data.vertices[face[2]][2])


    def calculate_model_matrix(self):
        self.model_matrix = Matrix44.identity()
        self.translation_matrix = Matrix44.from_translation(np.array([self.x, self.y, self.z]))
        self.scale_matrix = Matrix44.from_scale(np.array([self.scale_x, self.scale_y, self.scale_z]))
        self.model_matrix = self.model_matrix * self.translation_matrix * self.scale_matrix

    def move(self, x, y, z):
        if x == 0 and y == 0 and z == 0:
            return

        self.x += x
        self.y += y
        self.z += z
        self.calculate_model_matrix()

        return(self)

    def scale(self, x, y, z):
        if x == self.scale_x and y == self.scale_y and z == self.scale_z:
            return

        self.scale_x = x
        self.scale_y = y
        self.scale_z = z

        self.calculate_model_matrix()

        return(self)

    def scale_by(self, x, y):
        if x == 1 and y == 1 and z == 1:
            return

        self.scale(self.scale_x * x, self.scale_y * y, self.scale_z * z)

        return(self)

    def place(self, x, y, z):
        if x == self.x and y == self.y and z == self.z:
            return

        self.x = x
        self.y = y
        self.z = z
        self.calculate_model_matrix()

        return(self)


        