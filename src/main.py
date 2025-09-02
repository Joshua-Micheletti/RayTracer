"""main module."""

import time
from collections import deque
from typing import TYPE_CHECKING

import glfw
import numpy as np

from bvh.bvh_node import BVHNode, QBVHNode
from controller import Controller
from material import Material
from model import Model
from primitive import Box, Plane, Sphere, Triangle
from renderer import Renderer
from window.window import Window
import random
from scene import load_scene

if TYPE_CHECKING:
    from numpy.typing import NDArray

# import win_precise_time as wpt


def main() -> None:  # noqa: PLR0915
    """Main function."""
    window = Window.get_instance()
    controller = Controller.get_instance()
    renderer = Renderer.get_instance()

    load_scene("./scenes/test.toml")

    dt = 0

    last_frame_times = deque(maxlen=30)

    while not glfw.window_should_close(window.window):
        start = time.perf_counter()

        controller.update(dt)
        renderer.render()

        glfw.swap_buffers(window.window)
        glfw.poll_events()

        end = time.perf_counter()
        dt = end - start
        print(f"Total: {dt * 1000}")
        print(f"FPS: {1 / (dt if dt != 0 else 0.000001)}")
        print(f"Render: {renderer.render_time * 1000}")
        print(f"Samples: {renderer.rendered_frames}")
        print(f"Bounces: {renderer.bounces}")

        last_frame_times.append(dt * 1000)
        average = np.mean(list(last_frame_times))
        print("Average:", average)

    glfw.terminate()


if __name__ == "__main__":
    main()
