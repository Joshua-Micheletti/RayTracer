import numpy as np
from renderer.Renderer import Renderer
from model.Model import Model

class Data:

    __instance = None
    
    @staticmethod
    def getInstance():
        if Data.__instance == None:
            Data()
        return Data.__instance
    
    def __init__(self):
        if Data.__instance != None:
            raise Exception("Data already exists!")
        
        Data.__instance = self

        self.vertices = np.empty([0])
        self.vertices_sizes = np.empty([0])

        self.spheres = np.empty([0])
        self.sphere_colors = np.empty([0])

        self.planes = np.empty([0])
        self.plane_colors = np.empty([0])
        
        self.boxes = np.empty([0])
        self.boxes_colors = np.empty([0])

        self.model_matrixes = np.empty([0])
        self.colors = np.empty([0])
        self.models = np.empty([0])
        self.normals = np.empty([0])

        self.bounding_boxes = np.empty([0])

        self.materials = np.empty([0])
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


    def load_model(self, model, material_index = 0):
        self.models = np.append(self.models, model)
        self.load_vertices(model.vertices)
        self.load_model_mats(model.model_matrix)
        self.load_normals(model.normals)
        self.load_bounding_box(model.bounding_min, model.bounding_max)
        self.mesh_material_indices = np.append(self.mesh_material_indices, material_index)
        Renderer.getInstance().update_mesh_material_indices(self.mesh_material_indices)

    def load_sphere(self, center_x, center_y, center_z, radius, material_index = 0):
        self.spheres = np.append(self.spheres, np.array([center_x, center_y, center_z]))
        self.spheres = np.append(self.spheres, np.array([radius]))
        self.spheres = np.append(self.spheres, np.array([material_index]))
        # self.sphere_colors = np.append(self.sphere_colors, np.array([color_r, color_g, color_b, shininess]))

        Renderer.getInstance().update_spheres(self.spheres)
        # Renderer.getInstance().update_sphere_colors(self.sphere_colors)

    def load_plane(self, center_x, center_y, center_z, normal_x, normal_y, normal_z, material_index = 0):
        self.planes = np.append(self.planes, np.array([center_x, center_y, center_z, normal_x, normal_y, normal_z, material_index]))
        # self.plane_colors = np.append(self.plane_colors, np.array([color_r, color_g, color_b, shininess]))
        Renderer.getInstance().update_planes(self.planes)
        # Renderer.getInstance().update_plane_colors(self.plane_colors)

    def load_box(self, b0_x, b0_y, b0_z, b1_x, b1_y, b1_z, material_index = 0):
        self.boxes = np.append(self.boxes, np.array([b0_x, b0_y, b0_z, b1_x, b1_y, b1_z, material_index]))
        # self.boxes_colors = np.append(self.boxes_colors, np.array([color_r, color_g, color_b, shininess]))
        Renderer.getInstance().update_boxes(self.boxes)
        # Renderer.getInstance().update_boxes_colors(self.boxes_colors)


    def load_material(self, color_r, color_g, color_b, e_color_r, e_color_g, e_color_b, e_color_s, smoothness, specular_r, specular_g, specular_b, albedo):
        self.materials = np.append(self.materials, np.array([color_r, color_g, color_b, e_color_r, e_color_g, e_color_b, e_color_s, smoothness, specular_r, specular_g, specular_b, albedo]))
        Renderer.getInstance().update_materials(self.materials)


    def load_vertices(self, vertices):
        self.vertices = np.append(self.vertices, vertices)
        self.vertices_sizes = np.append(self.vertices_sizes, len(vertices))
        Renderer.getInstance().update_vertices(self.vertices)
        Renderer.getInstance().update_indices(self.vertices_sizes)

    def load_model_mats(self, model_mat):
        self.model_matrixes = np.append(self.model_matrixes, model_mat)
        Renderer.getInstance().update_model_mats(self.model_matrixes)

    def set_color(self, index, r, g, b, shininess):
        self.colors = np.append(self.colors, np.array([r, g, b, shininess]))
        Renderer.getInstance().update_colors(self.colors)

    def load_normals(self, normals):
        self.normals = np.append(self.normals, normals)
        Renderer.getInstance().update_normals(self.normals)

    def load_bounding_box(self, b0, b1):
        self.bounding_boxes = np.append(self.bounding_boxes, b0)
        self.bounding_boxes = np.append(self.bounding_boxes, b1)

        Renderer.getInstance().update_bounding_boxes(self.bounding_boxes)

    





        