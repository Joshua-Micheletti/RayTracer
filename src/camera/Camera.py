import numpy as np

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

        self.target = np.array([0.0, 0.0, 0.0])
        self.direction = normalize(self.position - self.target)

        self.up = np.array([0.0, 1.0, 0.0])
        self.right = normalize(np.cross(self.up, self.direction))

        self.camera_up = np.cross(self.direction, self.right)
        self.front = np.array([0.0, 0.0, -1.0])

        self.view_matrix = m3dLookAt(self.position, self.position + self.front, self.up)

    def get_view_matrix(self):
        self.view_matrix = m3dLookAt(self.position, self.position + self.front, self.up)
        return(self.view_matrix)


    def forward(self, amount):
        self.position += amount * self.front

    def move(self, amount):
        self.position += normalize(np.cross(self.front, self.up)) * amount


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

