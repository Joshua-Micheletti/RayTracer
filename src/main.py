# from window.Window import Window
# from controller.Controller import Controller
# import glfw

from renderer.Renderer import Renderer

def main():
    # window = Window.getInstance()
    # controller = Controller.getInstance()
    
    # while not glfw.window_should_close(window.window):
    #     controller.update()
        
    #     # Swap front and back buffers
    #     glfw.swap_buffers(window.window)
    #     # Poll for and process events
    #     glfw.poll_events()
        
    # glfw.terminate()
    
    renderer = Renderer.getInstance()
    renderer.render()


if __name__ == "__main__":
    main()