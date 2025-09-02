"""camera module."""

import math
from typing import cast

import numpy as np
import pyrr
from pyrr import Matrix44, Vector3

from utils.utils import normalize


class Camera:
    """
    Camera class.

    Raises:
        CameraInitializedError: Error in case the camera is initialized twice

    """

    __instance: "Camera | None" = None

    @staticmethod
    def get_instance() -> "Camera":
        """
        Singleton behavior.

        Returns:
            Camera: Instance of the Camera

        """
        if Camera.__instance is None:
            Camera()
        return cast("Camera", Camera.__instance)

    def __init__(self) -> None:
        """
        Initialization method.

        Raises:
            CameraInitializedError: Error in case the Camera is initialized more than once.

        """
        if Camera.__instance is not None:
            message: str = "Camera already exists!"
            raise CameraInitializedError(message)

        Camera.__instance = self

        self.position = np.array([8.0, 1.5, 0.0])
        self.world_up = np.array([0.0, 1.0, 0.0])
        self.up = np.array([0.0, 0.0, 0.0])
        self.front = np.array([0.0, 0.0, -1.0])
        self.right = np.array([0.0, 0.0, 0.0])

        self.yaw = 180.0
        self.pitch = 0

        self.projection_matrix = Matrix44.perspective_projection(45, 128 / 72, 0.1, 1)
        # self.projection_matrix = Matrix44.orthogonal_projection(-200, 200, -160, 160, -1, 1)

        self._update_camera_vectors()

        self.sensitivity = 0.1
        self.changed = False

    def get_view_matrix(self) -> Matrix44:
        """
        Get the view matrix.

        Returns:
            Matrix44: View matrix

        """
        self.view_matrix = self.look_at(
            self.position, self.position + self.front, np.array([0, 1, 0])
        )
        return self.view_matrix

    def get_inv_view_proj_matrix(self) -> Matrix44:
        """
        Get the inverse of the View Projection matrix.

        Returns:
            Matrix44: Inverse View Projection matrix

        """
        view_projection_matrix = self.projection_matrix * self.get_view_matrix()
        # view_projection_matrix = self.get_view_matrix()
        return ~view_projection_matrix

    def forward(self, amount: float) -> None:
        """
        Move the camera forward.

        Args:
            amount (float): Amount to move forward by

        """
        self.position += self.front * amount
        self._update_camera_vectors()

    def strafe(self, amount: float) -> None:
        """
        Method for moving the camera left or right.

        Args:
            amount (float): Amount to move by. Negative goes left, positive goes right

        """
        self.position += self.right * amount
        self._update_camera_vectors()

    def rise(self, amount: float) -> None:
        """
        Method for moving the camera on the world up vector.

        Args:
            amount (float): Amount to move the camera by

        """
        self.position += self.world_up * amount
        self._update_camera_vectors()

    def turn(self, x: int, y: int) -> None:
        """
        Method for turning the camera based on 2D offsets.

        Args:
            x (int): X movement
            y (int): Y movement

        """
        self.yaw += x * self.sensitivity
        self.pitch += y * self.sensitivity

        self.pitch = min(self.pitch, 89.0)
        self.pitch = max(self.pitch, -89.0)

        self._update_camera_vectors()

    def _update_camera_vectors(self) -> None:
        # self.front = np.array([0, 0, 0])
        fx = math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        fy = math.sin(math.radians(self.pitch))
        fz = math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        self.front = normalize(np.array([fx, fy, fz]))
        self.right = normalize(np.cross(self.front, self.world_up))
        self.up = np.cross(self.right, self.front)
        self.changed = True

    def look_at(
        self, position: list[float], target: list[float], world_up: list[float]
    ) -> Matrix44:
        """
        Create the look at matrix starting from a position, a target and the worldup vector.

        Args:
            position (list[float]): Viewer position
            target (list[float]): Viewer target
            world_up (list[float]): World up vector

        Returns:
            Matrix44: LookAt matrix

        """
        p_position = Vector3([position[0], position[1], position[2]])
        p_target = Vector3([target[0], target[1], target[2]])
        p_world_up = Vector3([world_up[0], world_up[1], world_up[2]])

        zaxis = pyrr.vector.normalise(p_position - p_target)
        xaxis = pyrr.vector.normalise(pyrr.vector3.cross(pyrr.vector.normalise(p_world_up), zaxis))
        yaxis = pyrr.vector3.cross(zaxis, xaxis)

        translation = pyrr.Matrix44.identity()
        translation[3][0] = -p_position.x
        translation[3][1] = -p_position.y
        translation[3][2] = -p_position.z

        rotation = pyrr.Matrix44.identity()
        rotation[0][0] = xaxis[0]
        rotation[1][0] = xaxis[1]
        rotation[2][0] = xaxis[2]
        rotation[0][1] = yaxis[0]
        rotation[1][1] = yaxis[1]
        rotation[2][1] = yaxis[2]
        rotation[0][2] = zaxis[0]
        rotation[1][2] = zaxis[1]
        rotation[2][2] = zaxis[2]

        return translation * rotation


class CameraInitializedError(Exception):
    """Custom exception for when the Camera is already initialized."""

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
