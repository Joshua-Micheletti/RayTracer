"""Module that defines the Window class."""

import glfw
import numpy as np

# from OpenGL.GL import *
from OpenGL import GL
from pyrr import Matrix44

from controller import Controller
from renderer import Renderer

first_mouse: bool = True


class Window:
    """Window class definition."""

    __instance: "Window | None" = None

    @staticmethod
    def get_instance() -> "Window":
        """
        Static method for getting an instance of the Window Singleton object.

        Returns:
            Window: Instance of the window object

        """
        if Window.__instance is None:
            Window()

        return Window.__instance

    def __init__(self, width: int = 1280, height: int = 720, name: str = "Pyllium") -> None:
        """
        Initialization method for the class.

        Args:
            width (int, optional): Initial width of the window. Defaults to 640.
            height (int, optional): Initial height of the window. Defaults to 480.
            name (str, optional): Name of the window. Defaults to "Pyllium".

        Raises:
            Exception: If the init is called when an instance already exists, it throws an exception

        """
        # if the window is already initalized, throw an exception
        if Window.__instance is not None:
            message: str = "Window already exists!"
            raise WindowInitializedError(message)

        # set the instance to the self object
        Window.__instance = self

        # initialize GLFW and return in case of an error
        if not glfw.init():
            return

        # setup the window hints for OpenGL
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.RED_BITS, 8)
        glfw.window_hint(glfw.GREEN_BITS, 8)
        glfw.window_hint(glfw.BLUE_BITS, 8)
        glfw.window_hint(glfw.DEPTH_BITS, 24)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        # glfw.window_hint(glfw.OPENGL_DEBUG_CONTEXT, glfw.FALSE)

        # initialize the window object using GLFW
        self.window = glfw.create_window(width, height, name, None, None)

        # if the window was not properly initialized, terminate GLFW and exit
        if not self.window:
            glfw.terminate()
            return

        # store the dimensions of the window
        self.width: int = width
        self.height: int = height

        # create the projection matrix as an orthogonal projection with the dimensions of the screen
        self.projection_matrix: Matrix44 = Matrix44.orthogonal_projection(
            -width / 2, width / 2, -height / 2, height / 2, -1, 1
        )

        # setup the callback functions for the window events
        glfw.set_key_callback(self.window, key_callback)
        glfw.set_framebuffer_size_callback(self.window, framebuffer_size_callback)
        glfw.set_cursor_pos_callback(self.window, mouse_callback)

        # setup the OpenGL context and set the vsync and input parameters
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)

        print("gl_compute_shader value: ", GL.GL_COMPUTE_SHADER)
        print("gl version", GL.glGetString(GL.GL_VERSION))
        num_exts = GL.glGetIntegerv(GL.GL_NUM_EXTENSIONS)
        for i in range(num_exts):
            print("ext:", GL.glGetStringi(GL.GL_EXTENSIONS, i).decode())

        params = [
            (GL.GL_MAX_COMPUTE_WORK_GROUP_COUNT, "GL_MAX_COMPUTE_WORK_GROUP_COUNT"),
            (GL.GL_MAX_COMPUTE_WORK_GROUP_SIZE, "GL_MAX_COMPUTE_WORK_GROUP_SIZE"),
            (GL.GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, "GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS"),
        ]

        for param, param_name in params:
            if param in (GL.GL_MAX_COMPUTE_WORK_GROUP_COUNT, GL.GL_MAX_COMPUTE_WORK_GROUP_SIZE):
                value = np.array([0, 0, 0], dtype=np.int32)
                GL.glGetIntegeri_v(param, 0, value)
                print(f"{param_name} (x): {value}")
            else:
                value = GL.glGetIntegerv(param)
                print(f"{param_name}: {value}")

        max_ubo_size = GL.glGetIntegerv(GL.GL_MAX_UNIFORM_BLOCK_SIZE)
        print("max_ubo_size", max_ubo_size)

        vendor = GL.glGetString(GL.GL_VENDOR).decode()
        renderer = GL.glGetString(GL.GL_RENDERER).decode()
        version = GL.glGetString(GL.GL_VERSION).decode()
        glsl_version = GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode()

        print(f"Vendor:   {vendor}")
        print(f"Renderer: {renderer}")
        print(f"OpenGL Version: {version}")
        print(f"GLSL Version:   {glsl_version}")


class WindowInitializedError(Exception):
    """Custom exception for when the Window is already initialized."""

    def __init__(self, message: str) -> None:
        """
        Initialization method.

        Args:
            message (str): Message to display in the error

        """
        self.message: str = message
        super().__init__(message)

    def __str__(self) -> str:
        """
        To string method.

        Returns:
            str: String representation of the object

        """
        return f"{self.message}"


def key_callback(window: any, key: str, _scancode: str, action: str, mods: str) -> None:
    """
    Callback function for when a key is pressed.

    Args:
        window (any): Window object
        key (str): Key that was pressed
        scancode (str): IDK
        action (str): Action that was performed
        mods (str): Modifiers active

    """
    # handle different type of key presses depending on the action value
    if action == glfw.PRESS:
        Controller.get_instance().handle_key_press(key, mods, window)
    if action == glfw.RELEASE:
        Controller.get_instance().handle_key_release(key, mods)


def framebuffer_size_callback(_window: any, width: int, height: int) -> None:
    """
    Callback function for when the window is resized.

    Args:
        window (any): Window object
        width (int): New window width
        height (int): New window height

    """
    # GL.glViewport(0, 0, width, height)

    # get a reference to the Window Singleton
    window_ref: Window = Window.get_instance()
    # update the window dimensions
    window_ref.width = width
    window_ref.height = height
    # update the window projection matrix
    window_ref.projection_matrix = Matrix44.orthogonal_projection(
        -width / 2, width / 2, -height / 2, height / 2, -1, 1
    )

    # update the renderer size
    Renderer.get_instance().update_size(width, height)


def mouse_callback(window: any, xpos: int, ypos: int) -> None:
    """
    Callback function for when the mouse moves.

    Args:
        window (any): Window object
        xpos (int): New x position of the mouse
        ypos (int): New y position of the mouse

    """
    Controller.get_instance().handle_mouse_movement(window, xpos, ypos)
