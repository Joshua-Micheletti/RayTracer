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
        self.model_matrixes = np.empty([0])
        self.models = np.empty([0])

    def move_light(self, x, y, z):
        self.models[2].move(x, y, z)

        for i in range(16):
            self.model_matrixes[2 * 16 - 1 + i] = self.models[2].model_matrix[int(i / 4)][3 - (i % 4)]

        # self.model_matrixes[2] = self.models[2].model_matrix

        Renderer.getInstance().update_model_mats(self.model_matrixes)
        Renderer.getInstance().light = np.array([self.models[2].x, self.models[2].y, self.models[2].z])

    def load_model(self, model):
        self.models = np.append(self.models, model)
        self.load_vertices(model.vertices)
        self.load_model_mats(model.model_matrix)

    def load_vertices(self, vertices):
        self.vertices = np.append(self.vertices, vertices)
        self.vertices_sizes = np.append(self.vertices_sizes, len(vertices))
        Renderer.getInstance().update_vertices(self.vertices)
        Renderer.getInstance().update_indices(self.vertices_sizes)

    def load_model_mats(self, model_mat):
        self.model_matrixes = np.append(self.model_matrixes, model_mat)
        Renderer.getInstance().update_model_mats(self.model_matrixes)

    





        