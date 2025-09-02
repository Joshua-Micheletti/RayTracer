"""model module."""

import numpy as np
import trimesh
from pyrr import Matrix44

from utils.utils import normalize


class Model:
    """Class to load and render a 3D mesh."""

    def __init__(self, filepath: str) -> None:
        """
        Initialization method.

        Args:
            filepath (str): Path of the model to load

        """
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

        # data = pywavefront.Wavefront(filepath, cache=True)
        data = trimesh.load(filepath)

        # Handle both Trimesh and Scene
        if isinstance(data, trimesh.Scene):
            meshes = data.geometry.values()
        else:
            meshes = [data]

        # Iterate through all meshes
        for mesh in meshes:
            # mesh.vertices[mesh.faces] gives shape (num_faces, 3, 3)
            triangles = mesh.vertices[mesh.faces]

            # Flatten triangles into self.vertices
            for tri in triangles:
                # tri is a 3x3 array: [[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]]
                for vertex in tri:
                    self.vertices.extend(vertex.tolist())

        # for v in data.vertices:
        #     x, y, z = v
        #     print(f"Vertex: ({x}, {y}, {z})")

        # for mesh in data.mesh_list:
        #     print("mesh", mesh)
        #     for face in mesh.faces:
        #         self.vertices.append(data.vertices[face[0]][0])
        #         self.vertices.append(data.vertices[face[0]][1])
        #         self.vertices.append(data.vertices[face[0]][2])

        #         self.vertices.append(data.vertices[face[1]][0])
        #         self.vertices.append(data.vertices[face[1]][1])
        #         self.vertices.append(data.vertices[face[1]][2])

        #         self.vertices.append(data.vertices[face[2]][0])
        #         self.vertices.append(data.vertices[face[2]][1])
        #         self.vertices.append(data.vertices[face[2]][2])

        # for name, mesh in data.meshes.items():
        #     for material in mesh.materials:
        #         self.vertices = material.vertices  # already flattened: [x,y,z, x,y,z, ...] per triangle
        #         # print(f"Mesh {name}, {len(verts)//3} vertices")

        # print("self.vertices", self.vertices)

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

    def calculate_model_matrix(self) -> None:
        """Private method for calculating the model matrix of the model."""
        self.model_matrix = Matrix44.identity()
        self.translation_matrix = Matrix44.from_translation(np.array([self.x, self.y, self.z]))
        self.scale_matrix = Matrix44.from_scale(
            np.array([self.scale_x, self.scale_y, self.scale_z])
        )
        self.model_matrix = self.model_matrix * self.translation_matrix * self.scale_matrix

    def move(self, x: float, y: float, z: float) -> "Model":
        """
        Move a model by a certain vector.

        Args:
            x (float): Movement on the x axis
            y (float): Movement on the y axis
            z (float): Movement on the z axis

        Returns:
            Model: self

        """
        if x == 0 and y == 0 and z == 0:
            return self

        self.x += x
        self.y += y
        self.z += z
        self.calculate_model_matrix()

        return self

    def scale(self, x: float, y: float, z: float) -> "Model":
        """
        Set the scale of the model.

        Args:
            x (float): Scale on the x axis
            y (float): Scale on the y axis
            z (float): Scale on the z axis

        Returns:
            Model: self

        """
        if x == self.scale_x and y == self.scale_y and z == self.scale_z:
            return self

        self.scale_x = x
        self.scale_y = y
        self.scale_z = z

        self.calculate_model_matrix()

        return self

    def scale_by(self, x: float, y: float, z: float) -> "Model":
        """
        Scale the model by the specified amounts starting from the current scale.

        Args:
            x (float): Scale multiplier on the x axis
            y (float): Scale multiplier on the y axis
            z (float): Scale multiplier on the z axis

        Returns:
            Model: self

        """
        if x == 1 and y == 1 and z == 1:
            return self

        self.scale(self.scale_x * x, self.scale_y * y, self.scale_z * z)

        return self

    def place(self, x: float, y: float, z: float) -> "Model":
        """
        Set the position of a model.

        Args:
            x (float): X coordinate
            y (float): Y coordinate
            z (float): Z coordinate

        Returns:
            Model: self

        """
        if x == self.x and y == self.y and z == self.z:
            return self

        self.x = x
        self.y = y
        self.z = z
        self.calculate_model_matrix()

        return self
