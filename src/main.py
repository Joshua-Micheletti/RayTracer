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

    data.load_material(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    data.load_material(0.2, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0)
    data.load_material(1.0, 0.2, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0)
    data.load_material(0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    data.load_material(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.5)
    data.load_material(0.0, 0.0, 0.0, 0.2, 1.0, 0.2, 2.0, 0.5)
    data.load_material(0.0, 0.0, 0.0, 1.0, 0.2, 0.2, 2.0, 0.5)
    data.load_material(0.0, 0.0, 0.0, 0.2, 0.2, 1.0, 2.0, 0.5)
    data.load_material(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0)

    gally = Model("../models/box.obj")
    tree = Model("../models/tree.obj")
    box = Model("../models/box.obj")

    tree.scale(0.04, 0.04, 0.04)
    tree.move(0, 0, 0)
    gally.move(5, 1, 0)
    gally.scale(0.1, 0.1, 0.1)
    box.scale(0.5, 0.5, 0.5)
    box.move(0, 1, 0)

    data.load_model(gally, 0)
    data.load_model(tree)
    data.load_model(box, 4)

    # data.set_color(0, 1.0, 1.0, 1.0, 1.0)
    # data.set_color(1, 1.0, 1.0, 0.5, 1.0)
    # data.set_color(2, 1.0, 1.0, 1.0, 1.0)

    data.load_sphere(0.0, -1, 0.0, 0.5, 8)
    # data.load_sphere(-1, 0.5, -3, 0.5, 1.0, 1.0, 1.0, 1.0)

    data.load_plane(0, 0, 0, 0, 1, 0, 5)
    data.load_plane(0, 0, 1, 0, 0, -1, 6)
    data.load_plane(0, 0, -1, 0, 0, 1, 7)
    data.load_plane(-1, 0, 0, 1, 0, 0)
    data.load_plane(1, 0, 0, -1, 0, 0)
    data.load_plane(0, 2, 0, 0, -1, 0)
    # data.load_plane(0, 100, 0, 0, -1, 0, 0.5, 0.5, 1.0)

    # data.load_box(-2, 2, 2, -1, 3, 3, 1.0, 1.0, 1.0, 1.0)

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
        print(f"Total: {dt * 1000}")
        print(f"FPS: {1 / (dt)}")
        print(f"Render: {renderer.render_time * 1000}")
        
    glfw.terminate()
    

if __name__ == "__main__":
    main()