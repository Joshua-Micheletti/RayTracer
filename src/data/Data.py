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

        self.to_update = True

    # def move_light(self, x, y, z):
    #     self.models[2].move(x, y, z)

    #     # for i in range(16):
    #     #     self.model_matrixes[2 * 16 - 1 + i] = self.models[2].model_matrix[int(i / 4)][3 - (i % 4)]

    #     Renderer.getInstance().update_model_mats(self.model_matrixes)
    #     Renderer.getInstance().light = np.array([self.models[2].x, self.models[2].y, self.models[2].z])
    #     Renderer.getInstance().light_model = self.models[2].model_matrix

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

    def load_vertices(self, vertices):
        self.vertices = np.append(self.vertices, vertices)
        self.vertices_sizes = np.append(self.vertices_sizes, len(vertices))
        Renderer.getInstance().update_vertices(self.vertices)
        Renderer.getInstance().update_indices(self.vertices_sizes)

    def load_model_mats(self, model_mat):
        self.model_matrixes = np.append(self.model_matrixes, model_mat)
        Renderer.getInstance().update_model_mats(self.model_matrixes)

    





        