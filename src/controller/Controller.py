import glfw
import time

class Controller:
    
    __instance = None
    
    @staticmethod
    def getInstance():
        if Controller.__instance == None:
            Controller()
        return Controller.__instance
    
    def __init__(self):
        if Controller.__instance != None:
            raise Exception("Controller already exists!")
        
        Controller.__instance = self
        
        self.states = dict()

        self.states["camera_left"] = False
        self.states["camera_right"] = False
        self.states["camera_up"] = False
        self.states["camera_down"] = False

        self.states["player_up"] = False
        self.states["player_down"] = False
        self.states["player_left"] = False
        self.states["player_right"] = False
        self.states["player_jumping"] = False

        self.states["display_bounding_box"] = False

        self.player_movement_speed = 1000
        self.player_jumping_strength = 1000
        self.camera_movement_speed = 40

        self.can_jump = True

        self.last_update = time.time()
        
        
    def handle_key_press(self, symbol, modifiers):
        if symbol == glfw.KEY_A:
            self.states["camera_left"] = True

        if symbol == glfw.KEY_S:
            self.states["camera_down"] = True

        if symbol == glfw.KEY_D:
            self.states["camera_right"] = True

        if symbol == glfw.KEY_W:
            self.states["camera_up"] = True


        if symbol == glfw.KEY_UP:
            self.states["player_up"] = True

        if symbol == glfw.KEY_DOWN:
            self.states["player_down"] = True

        if symbol == glfw.KEY_LEFT:
            self.states["player_left"] = True

        if symbol == glfw.KEY_RIGHT:
            self.states["player_right"] = True


        if symbol == glfw.KEY_SPACE:
            self.states["player_jumping"] = True


        if symbol == glfw.KEY_B and self.states["display_bounding_box"] == False:
            self.states["display_bounding_box"] = True

        elif symbol == glfw.KEY_B and self.states["display_bounding_box"] == True:
            self.states["display_bounding_box"] = False


    def handle_key_release(self, symbol, modifiers):
        if symbol == glfw.KEY_A:
            self.states["camera_left"] = False

        if symbol == glfw.KEY_S:
            self.states["camera_down"] = False

        if symbol == glfw.KEY_D:
            self.states["camera_right"] = False

        if symbol == glfw.KEY_W:
            self.states["camera_up"] = False


        if symbol == glfw.KEY_UP:
            self.states["player_up"] = False

        if symbol == glfw.KEY_DOWN:
            self.states["player_down"] = False

        if symbol == glfw.KEY_LEFT:
            self.states["player_left"] = False

        if symbol == glfw.KEY_RIGHT:
            self.states["player_right"] = False

        if symbol == glfw.KEY_SPACE:
            self.states["player_jumping"] = False
            
            
    def update(self):
        #print(self.states)
        pass