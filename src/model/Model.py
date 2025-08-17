import numpy as np
import pywavefront
from pyrr import Matrix44


class Model:

    def __init__(self, filepath):
        self.x = 0
        self.y = 0
        self.z = 0

        self.scale_x = 1
        self.scale_y = 1
        self.scale_z = 1

        self.bounding_min = np.array([0.0, 0.0, 0.0])
        self.bounding_max = np.array([0.0, 0.0, 0.0])

        self.model_matrix = Matrix44.identity()
        self.translation_matrix = Matrix44.identity()
        self.scale_matrix = Matrix44.identity()

        self.vertices = []
        self.normals = []

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

        for i in range(0, len(self.vertices), 9):
            v0 = np.array([self.vertices[i], self.vertices[i + 1], self.vertices[i + 2]])
            v1 = np.array([self.vertices[i + 3], self.vertices[i + 4], self.vertices[i + 5]])
            v2 = np.array([self.vertices[i + 6], self.vertices[i + 7], self.vertices[i + 8]])

            edge01 = v1 - v0
            edge02 = v2 - v0

            normal = normalize(np.cross(edge01, edge02))

            self.normals.append(normal[0])
            self.normals.append(normal[1])
            self.normals.append(normal[2])

        for i in range(0, len(self.vertices), 3):
            x = float(self.vertices[i])
            y = float(self.vertices[i + 1])
            z = float(self.vertices[i + 2])

            if i == 0:
                self.bounding_min[0] = x
                self.bounding_min[1] = y
                self.bounding_min[2] = z
                self.bounding_max[0] = x
                self.bounding_max[1] = y
                self.bounding_max[2] = z

            self.bounding_min[0] = min(x, self.bounding_min[0])

            self.bounding_max[0] = max(x, self.bounding_max[0])

            self.bounding_min[1] = min(y, self.bounding_min[1])

            self.bounding_max[1] = max(y, self.bounding_max[1])

            self.bounding_min[2] = min(z, self.bounding_min[2])

            self.bounding_max[2] = max(z, self.bounding_max[2])

            # print(x, y, z)
            # print(self.bounding_min)
            # print(self.bounding_max)

        # print(self.vertices)
        # print(self.bounding_min)
        # print(self.bounding_max)



    def calculate_model_matrix(self):
        self.model_matrix = Matrix44.identity()
        self.translation_matrix = Matrix44.from_translation(np.array([self.x, self.y, self.z]))
        self.scale_matrix = Matrix44.from_scale(np.array([self.scale_x, self.scale_y, self.scale_z]))
        self.model_matrix = self.model_matrix * self.translation_matrix * self.scale_matrix


    def move(self, x, y, z):
        if x == 0 and y == 0 and z == 0:
            return None

        self.x += x
        self.y += y
        self.z += z
        self.calculate_model_matrix()

        return(self)

    def scale(self, x, y, z):
        if x == self.scale_x and y == self.scale_y and z == self.scale_z:
            return None

        self.scale_x = x
        self.scale_y = y
        self.scale_z = z

        self.calculate_model_matrix()

        return(self)

    def scale_by(self, x, y):
        if x == 1 and y == 1 and z == 1:
            return None

        self.scale(self.scale_x * x, self.scale_y * y, self.scale_z * z)

        return(self)

    def place(self, x, y, z):
        if x == self.x and y == self.y and z == self.z:
            return None

        self.x = x
        self.y = y
        self.z = z
        self.calculate_model_matrix()

        return(self)


def normalize(vector):
    return vector / np.linalg.norm(vector)
