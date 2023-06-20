from window.Window import Window
from controller.Controller import Controller
import glfw

from renderer.Renderer import Renderer

def main():
    window = Window.getInstance()
    controller = Controller.getInstance()
    renderer = Renderer.getInstance()
    
    

    while not glfw.window_should_close(window.window):
        controller.update()
        
        renderer.render()

        # Swap front and back buffers
        glfw.swap_buffers(window.window)
        # Poll for and process events
        glfw.poll_events()
        
    glfw.terminate()
    
    
    


if __name__ == "__main__":
    main()