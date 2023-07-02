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


    def load_model(self, model):
        self.models = np.append(self.models, model)
        self.load_vertices(model.vertices)
        self.load_model_mats(model.model_matrix)
        self.load_normals(model.normals)

    def load_sphere(self, center_x, center_y, center_z, radius, color_r, color_g, color_b):
        self.spheres = np.append(self.spheres, np.array([center_x, center_y, center_z]))
        self.spheres = np.append(self.spheres, np.array([radius]))
        self.sphere_colors = np.append(self.sphere_colors, np.array([color_r, color_g, color_b]))
        Renderer.getInstance().update_spheres(self.spheres)
        Renderer.getInstance().update_sphere_colors(self.sphere_colors)

    def load_plane(self, center_x, center_y, center_z, normal_x, normal_y, normal_z, color_r, color_g, color_b):
        self.planes = np.append(self.planes, np.array([center_x, center_y, center_z, normal_x, normal_y, normal_z]))
        self.plane_colors = np.append(self.plane_colors, np.array([color_r, color_g, color_b]))
        Renderer.getInstance().update_planes(self.planes)
        Renderer.getInstance().update_plane_colors(self.plane_colors)

    def load_box(self, b0_x, b0_y, b0_z, b1_x, b1_y, b1_z, color_r, color_g, color_b):
        self.boxes = np.append(self.boxes, np.array([b0_x, b0_y, b0_z, b1_x, b1_y, b1_z]))
        self.boxes_colors = np.append(self.boxes_colors, np.array([color_r, color_g, color_b]))
        Renderer.getInstance().update_boxes(self.boxes)
        Renderer.getInstance().update_boxes_colors(self.boxes_colors)


    def load_vertices(self, vertices):
        self.vertices = np.append(self.vertices, vertices)
        self.vertices_sizes = np.append(self.vertices_sizes, len(vertices))
        Renderer.getInstance().update_vertices(self.vertices)
        Renderer.getInstance().update_indices(self.vertices_sizes)

    def load_model_mats(self, model_mat):
        self.model_matrixes = np.append(self.model_matrixes, model_mat)
        Renderer.getInstance().update_model_mats(self.model_matrixes)

    def set_color(self, index, r, g, b):
        self.colors = np.append(self.colors, [r, g, b])
        Renderer.getInstance().update_colors(self.colors)

    def load_normals(self, normals):
        self.normals = np.append(self.normals, normals)
        Renderer.getInstance().update_normals(self.normals)

    





        