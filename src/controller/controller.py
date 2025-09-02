"""controller module."""

import time
from typing import Any, cast

import glfw

from camera import Camera
from renderer import Renderer


class Controller:
    """
    Controller class.

    Raises:
        ControllerInitializedError: In case the object is initialized more than once.

    """

    __instance: "Controller | None" = None

    @staticmethod
    def get_instance() -> "Controller":
        """
        Singleton behavior.

        Returns:
            Controller: Controller instance

        """
        if Controller.__instance is None:
            Controller()
        return cast("Controller", Controller.__instance)

    def __init__(self) -> None:
        """
        Initialization method.

        Raises:
            ControllerInitializedError: Error in case the Controller is initialized more than once

        """
        if Controller.__instance is not None:
            message: str = "Controller already exists!"
            raise ControllerInitializedError(message)

        Controller.__instance = self

        self.states = {}

        self.states["camera_left"] = False
        self.states["camera_right"] = False
        self.states["camera_forward"] = False
        self.states["camera_backward"] = False
        self.states["camera_up"] = False
        self.states["camera_down"] = False

        self.states["display_bounding_box"] = False

        self.states["free_cursor"] = False

        self.states["accumulate"] = True

        self.states["denoise"] = False

        self.player_movement_speed = 1000
        self.player_jumping_strength = 1000
        self.camera_movement_speed = 40

        self.can_jump = True

        self.last_update = time.time()

        self.first_mouse = True
        self.lastx = 0
        self.lasty = 0

    def handle_key_press(self, symbol: int, _modifiers: int, window: Any) -> None:  # noqa: ANN401, C901, PLR0912
        """
        Handler for the key release event.

        Args:
            symbol (int): Key number that was released
            _modifiers (int): Active modifiers during the event
            window (Any): GLFW Window object

        """
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

        if symbol == glfw.KEY_B and not self.states["display_bounding_box"]:
            self.states["display_bounding_box"] = True

        elif symbol == glfw.KEY_B and self.states["display_bounding_box"]:
            self.states["display_bounding_box"] = False

        if symbol == glfw.KEY_LEFT_ALT and not self.states["free_cursor"]:
            self.states["free_cursor"] = True
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)

        elif symbol == glfw.KEY_LEFT_ALT and self.states["free_cursor"]:
            self.states["free_cursor"] = False
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

        if symbol == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, 1)

        if symbol == glfw.KEY_KP_SUBTRACT:
            Renderer.get_instance().bounces = Renderer.get_instance().bounces - 1

        if symbol == glfw.KEY_KP_ADD:
            Renderer.get_instance().bounces = Renderer.get_instance().bounces + 1

        if symbol == glfw.KEY_I:
            Renderer.get_instance().far_plane = Renderer.get_instance().far_plane + 0.01
        if symbol == glfw.KEY_K:
            Renderer.get_instance().far_plane = Renderer.get_instance().far_plane - 0.01

        if symbol == glfw.KEY_P and not self.states["denoise"]:
            Renderer.get_instance().denoise = 1
            self.states["denoise"] = True
        elif symbol == glfw.KEY_P and self.states["denoise"]:
            Renderer.get_instance().denoise = 0
            self.states["denoise"] = False

        if symbol == glfw.KEY_R and not self.states["accumulate"]:
            self.states["accumulate"] = True
        elif symbol == glfw.KEY_R and self.states["accumulate"]:
            self.states["accumulate"] = False

    def handle_key_release(self, symbol: int, _modifiers: int) -> None:
        """
        Handler for the key release event.

        Args:
            symbol (int): Key number that was released
            _modifiers (int): Active modifiers during the event

        """
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

    def handle_mouse_movement(self, _window: Any, x: int, y: int) -> None:  # noqa: ANN401
        """
        Method called when the mouse moves.

        Args:
            _window (Any): Window object
            x (int): New mouse X position
            y (int): New mouse Y position

        """
        if self.first_mouse:
            self.lastx = x
            self.lasty = y
            self.first_mouse = False

        xoffset = x - self.lastx
        yoffset = self.lasty - y

        self.lastx = x
        self.lasty = y

        Camera.get_instance().turn(xoffset, yoffset)

    def update(self, dt: float) -> None:
        """
        Update everything based on the internal states.

        Args:
            dt (float): Elapsed time since the last frame

        """
        if self.states["camera_forward"]:
            Camera.get_instance().forward(1 * dt)
        if self.states["camera_backward"]:
            Camera.get_instance().forward(-1 * dt)
        if self.states["camera_right"]:
            Camera.get_instance().strafe(1 * dt)
        if self.states["camera_left"]:
            Camera.get_instance().strafe(-1 * dt)
        if self.states["camera_up"]:
            Camera.get_instance().rise(1 * dt)
        if self.states["camera_down"]:
            Camera.get_instance().rise(-1 * dt)

        if not self.states["accumulate"]:
            Renderer.get_instance().reset_accumulation()


class ControllerInitializedError(Exception):
    """Custom exception for when the Controller is already initialized."""

    def __init__(self, message: str) -> None:
        """
        Initialization method.

        Args:
            message (str): Message to display in the error

        """
        self._message: str = message
        super().__init__(message)

    def __str__(self) -> str:
        """
        To string method.

        Returns:
            str: String representation of the object

        """
        return f"{self._message}"
