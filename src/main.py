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
    box.scale(0.2, 0.2, 0.2)

    data.load_model(gally)
    data.load_model(tree)
    data.load_model(box)

    data.set_color(0, 0.1, 0.1, 0.1)
    data.set_color(1, 1.0, 0.9, 0.1)
    data.set_color(2, 0.1, 0.1, 0.9)

    data.load_sphere(-1, 1, -1, 0.5, 1.0, 0.0, 0.0)

    data.load_plane(0, 0, 0, 0, 1, 0, 0, 1, 0)

    data.load_box(-2, 2, 2, -1, 3, 3, 1.0, 0.0, 1.0)

    dt = 0

    while not glfw.window_should_close(window.window):
        start = wpt.time()

        controller.update(dt)
        data.update()
        renderer.render()

        glfw.swap_buffers(window.window)
        glfw.poll_events()

        end = wpt.time()
        dt = end - start
        # print(f"Total: {dt * 1000}")
        # print(f"FPS: {1 / (dt)}")
        # print(f"Render: {renderer.render_time * 1000}")
        
    glfw.terminate()
    

if __name__ == "__main__":
    main()