from window.Window import Window
from controller.Controller import Controller
from data.Data import Data
from model.Model import Model
import glfw

from renderer.Renderer import Renderer

import win_precise_time as wpt

def main():
    window = Window.getInstance()
    controller = Controller.getInstance()
    renderer = Renderer.getInstance()
    data = Data.getInstance()

    gally = Model("../models/gally.obj")
    tree = Model("../models/tree.obj")
    box = Model("../models/box.obj")

    tree.scale(0.1, 0.1, 0.1)
    gally.move(5, 0, 0)
    box.scale(0.02, 0.02, 0.02)

    data.load_model(gally)
    data.load_model(tree)
    data.load_model(box)


    while not glfw.window_should_close(window.window):
        controller.update(0.06)
        renderer.render()

        glfw.swap_buffers(window.window)
        glfw.poll_events()
        
    glfw.terminate()
    

if __name__ == "__main__":
    main()