from window.Window import Window
from controller.Controller import Controller
from data.Data import Data
from model.Model import Model
import glfw

from renderer.Renderer import Renderer
import time
# import win_precise_time as wpt

def main():
    window = Window.getInstance()
    controller = Controller.getInstance()
    renderer = Renderer.getInstance()
    data = Data.getInstance()
    #                  color (r,g,b)  emission(rgb)  e_p shine albedo(r,g,b)  albedo
    data.load_material(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0) # 0 white opaque
    data.load_material(1.0, 0.2, 0.2, 0.0, 0.0, 0.0, 1.0, 0.2, 1.0, 0.2, 0.2, 1.0) # 1 red 0.2 shine
    data.load_material(0.2, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.4, 0.2, 1.0, 0.2, 1.0) # 2 green 0.4 shine
    data.load_material(0.2, 0.2, 1.0, 0.0, 0.0, 0.0, 1.0, 0.6, 0.2, 0.2, 1.0, 1.0) # 3 blue 0.6 shine
    data.load_material(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.8, 1.0, 1.0, 1.0, 1.0) # 4 white 0.8 shine
    data.load_material(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0) # 5 white 1.0 shine
    data.load_material(1.0, 0.2, 0.2, 0.0, 0.0, 0.0, 3.0, 0.5, 1.0, 1.0, 1.0, 0.2) # 6 red shiny albedo 20%
    data.load_material(0.2, 1.0, 0.2, 0.0, 0.0, 0.0, 3.0, 1.0, 1.0, 1.0, 1.0, 0.5) # 7 green shiny albedo 50%
    data.load_material(0.2, 0.2, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0) # 8 blue shiny albedo 100%
    data.load_material(1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 0.5, 1.0, 1.0, 1.0, 0.0) # 9 emissive red
    data.load_material(1.0, 1.0, 1.0, 0.2, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0) # 10 emissive blue
    data.load_material(1.0, 0.2, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0) # 11 red
    data.load_material(0.2, 0.2, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0) # 12 blue
    data.load_material(0.2, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0) # 13 green
    data.load_material(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 1.0, 1.0, 0.0) # 14 emissive white
    data.load_material(124 / 255, 112 / 255, 208 / 255, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0) # 15 white opaque
    # rgb(124, 112, 208)
    

    gally = Model("../models/gally.obj")
    tree = Model("../models/tree.obj")
    box = Model("../models/box.obj")

    tree.scale(0.04, 0.04, 0.04)
    tree.move(0, 0, 0)
    gally.move(0, 0.5, 0)
    gally.scale(0.1, 0.1, 0.1)
    box.scale(0.4, 0.4, 0.4)
    box.move(-0.5, 2.30, 0)

    data.load_model(gally, 15)
    # data.load_model(tree, 10)
    # data.load_model(box, 14)

    # data.set_color(0, 1.0, 1.0, 1.0, 1.0)
    # data.set_color(1, 1.0, 1.0, 0.5, 1.0)
    # data.set_color(2, 1.0, 1.0, 1.0, 1.0)

    # data.load_sphere(0.0, 0.25, 0.0, 0.25)
    # data.load_sphere(0.0, 0.25, -0.75, 0.25, 9)
    # data.load_sphere(0.0, 0.25, 0.75, 0.25, 10)
    data.load_sphere(-0.80, 1, -0.80, 0.15, 0)
    data.load_sphere(-0.80, 1, -0.40, 0.15, 1)
    data.load_sphere(-0.80, 1, 0.0, 0.15, 2)
    data.load_sphere(-0.80, 1, 0.40, 0.15, 3)
    data.load_sphere(-0.80, 1, 0.80, 0.15, 4)
    data.load_sphere(-0.40, 0.5, -0.80, 0.15, 5)
    data.load_sphere(-0.40, 0.5, -0.40, 0.15, 6)
    data.load_sphere(-0.40, 0.5, 0.0, 0.15, 7)
    data.load_sphere(-0.40, 0.5, 0.40, 0.15, 8)
    data.load_sphere(-0.40, 0.5, 0.80, 0.15, 9)
    data.load_sphere(-0.60, 0.75, 0.0, 0.15, 10)
    data.load_sphere(2, 3.0, 0, 2, 14)
    # data.load_sphere(-50, 50, -0.0, 10, 4)
    # data.load_sphere(-0.80, 1,  0.40, 0.15, 8)
    # data.load_sphere(-0.80, 1,  0.80, 0.15, 8)
    # data.load_sphere(-1, 0.5, -3, 0.5, 1.0, 1.0, 1.0, 1.0)

    data.load_plane(0, 0, 0, 0, 1, 0, 13)
    data.load_plane(0, 0, 1, 0, 0, -1, 12)
    data.load_plane(0, 0, -1, 0, 0, 1, 11)
    data.load_plane(-1, 0, 0, 1, 0, 0)
    data.load_plane(1, 0, 0, -1, 0, 0)
    data.load_plane(0, 2, 0, 0, -1, 0)
    # data.load_plane(0, 100, 0, 0, -1, 0, 0.5, 0.5, 1.0)

    # data.load_box(-2, 2, 2, -1, 3, 3, 1.0, 1.0, 1.0, 1.0)

    dt = 0

    while not glfw.window_should_close(window.window):
        start = time.time()

        controller.update(dt)
        data.update()
        renderer.render()

        glfw.swap_buffers(window.window)
        glfw.poll_events()

        end = time.time()
        dt = end - start
        print(f"Total: {dt * 1000}")
        print(f"FPS: {1 / (dt)}")
        print(f"Render: {renderer.render_time * 1000}")
        print(f"Samples: {renderer.rendered_frames}")
        print(f"Bounces: {renderer.bounces}")
        
    glfw.terminate()
    

if __name__ == "__main__":
    main()