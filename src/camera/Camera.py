import numpy as np
import math
from pyrr import Matrix44, Vector3
import pyrr

class Camera:
    __instance = None
    
    @staticmethod
    def getInstance():
        if Camera.__instance == None:
            Camera()
        return Camera.__instance
    
    def __init__(self):
        if Camera.__instance != None:
            raise Exception("Window already exists!")
        
        Camera.__instance = self

        self.position = np.array([0.0, 0.0, 0.0])
        self.world_up = np.array([0.0, 1.0, 0.0])
        self.up = np.array([0.0, 0.0, 0.0])
        self.front = np.array([0.0, 0.0, -1.0])
        self.right = np.array([0.0, 0.0, 0.0])

        self.yaw = -90.0
        self.pitch = 0

        self.projection_matrix = Matrix44.perspective_projection(45, 192 / 108, 0.1, 1)
        # self.projection_matrix = Matrix44.orthogonal_projection(-200, 200, -160, 160, -1, 1)

        self.update_camera_vectors()

        self.sensitivity = 0.1

    def get_view_matrix(self):
        self.view_matrix = self.look_at(self.position, self.position + self.front, np.array([0, 1, 0]))
        return(self.view_matrix)

    def get_inv_view_proj_matrix(self):
        view_projection_matrix = self.projection_matrix * self.get_view_matrix()
        # view_projection_matrix = self.get_view_matrix()
        return(~view_projection_matrix)


    def forward(self, amount):
        self.position += self.front * amount
        self.update_camera_vectors()

    def strafe(self, amount):
        self.position += self.right * amount
        self.update_camera_vectors()

    def rise(self, amount):
        self.position += self.world_up * amount
        self.update_camera_vectors()

    def turn(self, x, y):
        self.yaw   += x * self.sensitivity
        self.pitch += y * self.sensitivity

        if self.pitch > 89.0:
            self.pitch = 89.0
        if self.pitch < -89.0:
            self.pitch = -89.0

        self.update_camera_vectors()

    def update_camera_vectors(self):
        #self.front = np.array([0, 0, 0])
        fx = math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        fy = math.sin(math.radians(self.pitch))
        fz = math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        self.front = normalize(np.array([fx, fy, fz]))
        self.right = normalize(np.cross(self.front, self.world_up))
        self.up = np.cross(self.right, self.front)

    def look_at(self, position, target, world_up):
        pPosition = Vector3([position[0], position[1], position[2]])
        pTarget = Vector3([target[0], target[1], target[2]])
        pWorld_up = Vector3([world_up[0], world_up[1], world_up[2]])

        zaxis = pyrr.vector.normalise(pPosition - pTarget)
        xaxis = pyrr.vector.normalise(pyrr.vector3.cross(pyrr.vector.normalise(pWorld_up), zaxis))
        yaxis = pyrr.vector3.cross(zaxis, xaxis)

        translation = pyrr.Matrix44.identity()
        translation[3][0] = -pPosition.x
        translation[3][1] = -pPosition.y
        translation[3][2] = -pPosition.z

        rotation = pyrr.Matrix44.identity()
        rotation[0][0] = xaxis[0]
        rotation[1][0] = xaxis[1]
        rotation[2][0] = xaxis[2]
        rotation[0][1] = yaxis[0]
        rotation[1][1] = yaxis[1]
        rotation[2][1] = yaxis[2]
        rotation[0][2] = zaxis[0]
        rotation[1][2] = zaxis[1]
        rotation[2][2] = zaxis[2]

        return(translation * rotation)

def m3dLookAt(eye, target, up):
    mz = normalize( (eye[0]-target[0], eye[1]-target[1], eye[2]-target[2]) ) # inverse line of sight
    mx = normalize(np.cross( up, mz ) )
    my = normalize(np.cross( mz, mx ) )
    tx =  np.dot( mx, eye )
    ty =  np.dot( my, eye )
    tz = -np.dot( mz, eye )   
    return np.array([mx[0], my[0], mz[0], 0, mx[1], my[1], mz[1], 0, mx[2], my[2], mz[2], 0, tx, ty, tz, 1])

def normalize(vector):
    return vector / np.linalg.norm(vector)

