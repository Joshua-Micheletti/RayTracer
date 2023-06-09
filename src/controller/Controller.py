import glfw
import time
from renderer.Renderer import Renderer
from camera.Camera import Camera
from data.Data import Data

first_mouse = True
lastx = 0
lasty = 0

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
        self.states["camera_forward"] = False
        self.states["camera_backward"] = False
        self.states["camera_up"] = False
        self.states["camera_down"] = False

        self.states["player_up"] = False
        self.states["player_down"] = False
        self.states["player_left"] = False
        self.states["player_right"] = False
        self.states["player_forward"] = False
        self.states["player_backward"] = False
        self.states["player_jumping"] = False

        self.states["display_bounding_box"] = False

        self.states["free_cursor"] = False

        self.states["accumulate"] = False

        self.states["denoise"] = False

        self.player_movement_speed = 1000
        self.player_jumping_strength = 1000
        self.camera_movement_speed = 40

        self.can_jump = True

        self.last_update = time.time()
        
        
    def handle_key_press(self, symbol, modifiers, window):
        if symbol == glfw.KEY_A:
            self.states["camera_left"] = True

        if symbol == glfw.KEY_S:
            self.states["camera_backward"] = True

        if symbol == glfw.KEY_D:
            self.states["camera_right"] = True

        if symbol == glfw.KEY_W:
            self.states["camera_forward"] = True

        if symbol == glfw.KEY_SPACE:
            self.states["camera_up"] = True

        if symbol == glfw.KEY_LEFT_CONTROL:
            self.states["camera_down"] = True


        if symbol == glfw.KEY_UP:
            self.states["player_up"] = True

        if symbol == glfw.KEY_DOWN:
            self.states["player_down"] = True

        if symbol == glfw.KEY_LEFT:
            self.states["player_left"] = True

        if symbol == glfw.KEY_RIGHT:
            self.states["player_right"] = True

        if symbol == glfw.KEY_E:
            self.states["player_forward"] = True

        if symbol == glfw.KEY_Q:
            self.states["player_backward"] = True


        if symbol == glfw.KEY_B and self.states["display_bounding_box"] == False:
            self.states["display_bounding_box"] = True

        elif symbol == glfw.KEY_B and self.states["display_bounding_box"] == True:
            self.states["display_bounding_box"] = False


        if symbol == glfw.KEY_LEFT_ALT and self.states["free_cursor"] == False:
            self.states["free_cursor"] = True
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL) 

        elif symbol == glfw.KEY_LEFT_ALT and self.states["free_cursor"] == True:
            self.states["free_cursor"] = False
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED) 


        if symbol == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, 1)

        if symbol == glfw.KEY_KP_SUBTRACT:
            Renderer.getInstance().bounces = Renderer.getInstance().bounces - 1
        
        if symbol == glfw.KEY_KP_ADD:
            Renderer.getInstance().bounces = Renderer.getInstance().bounces + 1

        if symbol == glfw.KEY_I:
            Renderer.getInstance().far_plane = Renderer.getInstance().far_plane + 0.01
        if symbol == glfw.KEY_K:
            Renderer.getInstance().far_plane = Renderer.getInstance().far_plane - 0.01

        if symbol == glfw.KEY_P and self.states["denoise"] == False:
            Renderer.getInstance().denoise = 1
            self.states["denoise"] = True
        elif symbol == glfw.KEY_P and self.states["denoise"] == True:
            Renderer.getInstance().denoise = 0
            self.states["denoise"] = False

        if symbol == glfw.KEY_R and self.states["accumulate"] == False:
            self.states["accumulate"] = True
        elif symbol == glfw.KEY_R and self.states["accumulate"] == True:
            self.states["accumulate"] = False    
        


    def handle_key_release(self, symbol, modifiers):
        if symbol == glfw.KEY_A:
            self.states["camera_left"] = False

        if symbol == glfw.KEY_S:
            self.states["camera_backward"] = False

        if symbol == glfw.KEY_D:
            self.states["camera_right"] = False

        if symbol == glfw.KEY_W:
            self.states["camera_forward"] = False

        if symbol == glfw.KEY_SPACE:
            self.states["camera_up"] = False

        if symbol == glfw.KEY_LEFT_CONTROL:
            self.states["camera_down"] = False


        if symbol == glfw.KEY_UP:
            self.states["player_up"] = False

        if symbol == glfw.KEY_DOWN:
            self.states["player_down"] = False

        if symbol == glfw.KEY_LEFT:
            self.states["player_left"] = False

        if symbol == glfw.KEY_RIGHT:
            self.states["player_right"] = False

        if symbol == glfw.KEY_E:
            self.states["player_forward"] = False

        if symbol == glfw.KEY_Q:
            self.states["player_backward"] = False
            

    def handle_mouse_movement(self, window, x, y):
        global first_mouse
        global lastx
        global lasty

        if (first_mouse):
            lastx = x
            lasty = y
            first_mouse = False

        xoffset = x - lastx
        yoffset = lasty - y

        lastx = x
        lasty = y

        Camera.getInstance().turn(xoffset, yoffset)


            
    def update(self, dt):
        player_target = 2

        if self.states["player_up"]:
            Data.getInstance().move_model(player_target, 0 * dt, 1*dt, 0*dt)
        if self.states["player_down"]:
            Data.getInstance().move_model(player_target, 0*dt, -1*dt, 0*dt)
        if self.states["player_left"]:
            Data.getInstance().move_model(player_target, -1*dt, 0*dt, 0*dt)
        if self.states["player_right"]:
            Data.getInstance().move_model(player_target, 1*dt, 0*dt, 0*dt)
        if self.states["player_forward"]:
            Data.getInstance().move_model(player_target, 0*dt, 0 * dt, 1*dt)
        if self.states["player_backward"]:
            Data.getInstance().move_model(player_target, 0*dt, 0*dt, -1*dt)

        if self.states["camera_forward"]:
            Camera.getInstance().forward(1 * dt)
        if self.states["camera_backward"]:
            Camera.getInstance().forward(-1 * dt)
        if self.states["camera_right"]:
            Camera.getInstance().strafe(1 * dt)
        if self.states["camera_left"]:
            Camera.getInstance().strafe(-1 * dt)
        if self.states["camera_up"]:
            Camera.getInstance().rise(1 * dt)
        if self.states["camera_down"]:
            Camera.getInstance().rise(-1 * dt)

        if self.states["accumulate"] == False:
            Renderer.getInstance().reset_accumulation()