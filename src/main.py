from window.Window import Window
from controller.Controller import Controller
import glfw

from renderer.Renderer import Renderer

import win_precise_time as wpt

def main():
    window = Window.getInstance()
    controller = Controller.getInstance()
    renderer = Renderer.getInstance()
    
    # printTick = 1 / 1
    # printClock = wpt.time()

    while not glfw.window_should_close(window.window):
        # current_time = wpt.time()
        
        # if dt > printTick:
        #     print(f"GameLoop: {round(exec_time * 1000, 2)}")
        #     print(f"Rendering: {round(renderer.render_time * 1000, 2)}")
        #     printClock += printTick

        renderer.render()
        # Swap front and back buffers
        glfw.swap_buffers(window.window)
        # renderClock += tick

        controller.update(0.16)

        # Poll for and process events
        glfw.poll_events()

        # end_time = wpt.time()
        

        
    glfw.terminate()
    

if __name__ == "__main__":
    main()