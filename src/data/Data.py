import numpy as np

from primitive.sphere import Sphere
from renderer import Renderer


class Data:
    sphere_dtype = np.dtype(
        [
            ("center", np.float32, 3),  # 3 floats (vec3)
            ("radius", np.float32),
            ("material", np.int32),
            ("_pad", np.float32, 3),  # padding float to make vec3 align as vec4
        ],
        align=True,
    )

    plane_dtype = np.dtype(
        [
            ("center", np.float32, 3),  # 12B
            ("material", np.int32),  # 4B
            ("normal", np.float32, 3),  # 12B
            ("_pad", np.int32),  # 4B
        ],
        align=True,
    )

    bvh_node_dtype = np.dtype(
        [
            ("aabb_min", np.float32, 3),  # vec3 + padding
            ("left", np.int32),
            ("aabb_max", np.float32, 3),  # vec3 + padding
            ("right", np.int32),
            ("primitive", np.int32),
            ("primitive_type", np.int32),
            ("_padding", np.int32, 2),  # to align struct size to 16 bytes multiple
        ],
        align=True,
    )

    __instance = None

    @staticmethod
    def getInstance():
        if Data.__instance is None:
            Data()
        return Data.__instance

    def __init__(self):
        if Data.__instance is not None:
            raise Exception("Data already exists!")

        Data.__instance = self

        self.renderer: Renderer = Renderer.get_instance()

        self.vertices = np.empty([0])
        self.vertices_sizes = np.empty([0])

        self.spheres = np.zeros(0, dtype=self.sphere_dtype)
        self.sphere_colors = np.empty([0])

        self.planes = np.empty(0, dtype=self.plane_dtype)
        self.plane_colors = np.empty([0])

        self.boxes = np.zeros(0, dtype=self.plane_dtype)
        self.boxes_colors = np.empty([0])

        self.model_matrixes = np.empty([0])
        self.colors = np.empty([0])
        self.models = np.empty([0])
        self.normals = np.empty([0])

        self.bounding_boxes = np.empty([0])

        self.materials = np.zeros(0, dtype=np.float32)
        self.mesh_material_indices = np.empty([0])

        self.to_update = True

    def update(self):
        if self.to_update:
            self.to_update = False

            self.model_matrixes = np.empty([0])

            for i in range(len(self.models)):
                self.load_model_mats(self.models[i].model_matrix)
                if i == 2:
                    Renderer.getInstance().light_model = self.models[i].model_matrix

    def move_model(self, index, x, y, z):
        self.to_update = True
        self.models[index].move(x, y, z)

    def load_model(self, model, material_index=0):
        self.models = np.append(self.models, model)
        self.load_vertices(model.vertices)
        self.load_model_mats(model.model_matrix)
        self.load_normals(model.normals)
        self.load_bounding_box(model.bounding_min, model.bounding_max)
        self.mesh_material_indices = np.append(
            self.mesh_material_indices, material_index,
        )
        self.renderer.update_ssbo(
            self.renderer.mesh_material_indices, self.mesh_material_indices, 2,
        )

    def load_sphere(self, center_x, center_y, center_z, radius, material_index=0):
        new_sphere = np.zeros(1, dtype=self.sphere_dtype)
        new_sphere[0]["center"] = [center_x, center_y, center_z]
        new_sphere[0]["radius"] = radius
        new_sphere[0]["material"] = material_index
        new_sphere[0]["_pad"] = [0.0, 0.0, 0.0]  # padding float

        self.spheres = np.concatenate([self.spheres, new_sphere])

        self.renderer.update_ssbo(self.renderer.spheres, self.spheres, 5)

    def load_plane(
        self,
        center_x,
        center_y,
        center_z,
        normal_x,
        normal_y,
        normal_z,
        material_index=0,
    ):
        print("material_index", material_index)
        new_plane = np.zeros(1, dtype=self.plane_dtype)
        new_plane[0]["center"] = [center_x, center_y, center_z]
        new_plane[0]["material"] = material_index
        new_plane[0]["normal"] = [normal_x, normal_y, normal_z]
        new_plane[0]["_pad"] = 0.0  # padding float

        self.planes = np.concatenate([self.planes, new_plane])

        self.renderer.update_ssbo(self.renderer.planes, self.planes, 6)

    def load_box(self, b0_x, b0_y, b0_z, b1_x, b1_y, b1_z, material_index=0):
        self.boxes = np.append(
            self.boxes, np.array([b0_x, b0_y, b0_z, b1_x, b1_y, b1_z, material_index]),
        )
        self.renderer.update_ssbo(self.renderer.boxes, self.boxes, 7)

    def load_bvh(self, flat_nodes):
        n = len(flat_nodes)
        np_bvh = np.zeros(n, dtype=self.bvh_node_dtype)

        for i, node in enumerate(flat_nodes):
            # aabb_min and aabb_max need 4 floats (vec3 + padding)
            # So pack with a zero at the end for padding
            np_bvh[i]["aabb_min"] = node["aabb_min"]
            # np_bvh[i]["aabb_min"][3] = 0.0  # padding

            np_bvh[i]["aabb_max"] = node["aabb_max"]
            # np_bvh[i]["aabb_max"][3] = 0.0  # padding

            np_bvh[i]["left"] = node["left"]
            np_bvh[i]["right"] = node["right"]
            np_bvh[i]["primitive"] = node["primitive"]
            np_bvh[i]["primitive_type"] = node["primitive_type"]
            np_bvh[i]["_padding"] = [0, 0]  # keep zero

        print(np_bvh)

        self.renderer.update_ssbo(self.renderer.bvh, np_bvh, 11)

    def load_material(
        self,
        color_r,
        color_g,
        color_b,
        e_color_r,
        e_color_g,
        e_color_b,
        e_color_s,
        smoothness,
        specular_r,
        specular_g,
        specular_b,
        albedo,
    ):
        self.materials = np.append(
            self.materials,
            np.array(
                [
                    color_r,
                    color_g,
                    color_b,
                    e_color_r,
                    e_color_g,
                    e_color_b,
                    e_color_s,
                    smoothness,
                    specular_r,
                    specular_g,
                    specular_b,
                    albedo,
                ],
                dtype=np.float32,
            ),
        )
        self.renderer.update_ssbo(self.renderer.materials, self.materials, 9)

    def load_vertices(self, vertices):
        self.vertices = np.append(self.vertices, vertices)
        self.vertices_sizes = np.append(self.vertices_sizes, len(vertices))
        # Renderer.getInstance().update_vertices(self.vertices)
        # Renderer.getInstance().update_indices(self.vertices_sizes)

    def load_model_mats(self, model_mat):
        self.model_matrixes = np.append(self.model_matrixes, model_mat)
        # Renderer.getInstance().update_model_mats(self.model_matrixes)

    def set_color(self, index, r, g, b, shininess):
        self.colors = np.append(self.colors, np.array([r, g, b, shininess]))
        # Renderer.getInstance().update_colors(self.colors)

    def load_normals(self, normals):
        self.normals = np.append(self.normals, normals)
        # Renderer.getInstance().update_normals(self.normals)

    def load_bounding_box(self, b0, b1):
        self.bounding_boxes = np.append(self.bounding_boxes, b0)
        self.bounding_boxes = np.append(self.bounding_boxes, b1)

        self.renderer.update_ssbo(self.renderer.bounding_boxes, self.bounding_boxes, 8)

    def load_primitive(self, primitive) -> int:
        index: int = -1

        if (isinstance(primitive, Sphere)):
            new_sphere = np.zeros(1, dtype=self.sphere_dtype)
            new_sphere[0]["center"] = [primitive.center[0], primitive.center[1], primitive.center[2]]
            new_sphere[0]["radius"] = primitive.radius
            new_sphere[0]["material"] = primitive.material
            new_sphere[0]["_pad"] = [0.0, 0.0, 0.0]  # padding float

            index = len(self.spheres)

            self.spheres = np.concatenate([self.spheres, new_sphere])

            self.renderer.update_ssbo(self.renderer.spheres, self.spheres, 5)

        return index
