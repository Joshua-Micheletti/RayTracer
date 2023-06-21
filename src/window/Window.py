import glfw
from pyrr import Matrix44
from controller.Controller import Controller
from OpenGL.GL import *


class Window():
    
    __instance = None
    
    @staticmethod
    def getInstance():
        if Window.__instance == None:
            Window()
        return Window.__instance
    
    def __init__(self, width = 200, height = 160, name = "Pyllium"):
        if Window.__instance != None:
            raise Exception("Window already exists!")
        
        Window.__instance = self
        
        if not glfw.init():
            return

        self.window = glfw.create_window(width, height, name, None, None)

        if not self.window:
            glfw.terminate()
            return

        self.projection_matrix = Matrix44.orthogonal_projection(-width/2, width/2, -height/2, height/2, -1, 1)

        self.width = width
        self.height = height

        glfw.set_key_callback(self.window, key_callback);
        glfw.set_framebuffer_size_callback(self.window, framebuffer_size_callback);

        glfw.make_context_current(self.window)
        glfw.swap_interval(0);

        


def key_callback(window, key, scancode, action, mods):
    if action == glfw.PRESS:
        Controller.getInstance().handle_key_press(key, mods)
    if action == glfw.RELEASE:
        Controller.getInstance().handle_key_release(key, mods)


def framebuffer_size_callback(window, width, height):
    glViewport(0, 0, width, height)
    Window.getInstance().width = width
    Window.getInstance().height = height
    Window.getInstance().projection_matrix = Matrix44.orthogonal_projection(-width/2, width/2, -height/2, height/2, -1, 1)