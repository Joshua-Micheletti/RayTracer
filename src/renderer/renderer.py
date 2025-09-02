"""Renderer module."""

# ruff: noqa: F403, F405
from __future__ import annotations

import time
from ctypes import *
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from OpenGL import GL

from camera import Camera
from shader import Shader
from utils.opengl_utils import create_screen_quad_vao
from utils.typecheck import is_positive_int

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Renderer:
    """Renderer class."""

    # renderer dimensions
    width: int
    """Width of the renderer"""

    height: int
    """Height of the renderer"""

    # buffers
    vertices: int
    model_mats: int
    indices: int
    normals: int
    spheres: int
    planes: int
    boxes: int
    bounding_boxes: int
    materials: int
    mesh_material_indices: int
    triangles: int
    bvh: int

    # camera object
    camera: Camera

    # Shaders
    compute_shader: int
    program_id: int
    screen_shader: Shader

    # render parameters
    render_time: float
    rendered_frames: int
    bounces: int
    denoise: int
    far_plane: float

    # static reference
    __instance: Renderer | None = None

    def __init__(self) -> None:
        """Init function, it sets up all the requried components for the renderer."""
        # Handle the class as a Singleton
        if Renderer.__instance is not None:
            message: str = "Renderer already exists!"
            raise RendererInitializedError(message)

        # store the instance upon creation
        Renderer.__instance = self

        # setup the dimensions of the renderer
        self._width = 1280
        self._height = 720

        self._scaling_factor = 1.0

        self._frames = 0

        self.scene_min_coords = []

        # set the clear color for clearing the buffer
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)
        # enable alpha blending
        GL.glEnable(GL.GL_BLEND)
        # define the blending functions
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        # setup the vertices, VBO and VAO for the screen quad for rendering
        self._vao = create_screen_quad_vao()

        # setup the screen textures
        self._texture = self._setup_screen_texture(GL.GL_TEXTURE0)
        self._last_frame = self._setup_screen_texture(GL.GL_TEXTURE1)

        # create all the buffers to store all the data to send the shader
        self.spheres = GL.glGenBuffers(1)
        self.planes = GL.glGenBuffers(1)
        self.boxes = GL.glGenBuffers(1)
        self.materials = GL.glGenBuffers(1)
        self.bvh = GL.glGenBuffers(1)
        self.triangles = GL.glGenBuffers(1)

        # setup the camera
        self._camera = Camera.get_instance()

        # prepare the render parameters
        self._render_time = 0
        self.rendered_frames = 0
        self._bounces = 10
        self._denoise = 0
        self._far_plane = 1.0

        # create the compute shader and program
        self._compute_shader: int = GL.glCreateShader(GL.GL_COMPUTE_SHADER)
        self._program_id: int = GL.glCreateProgram()

        # read the source of the compute shader
        compute_shader_source: str = ""
        with Path("./shaders/compute.glsl").open("+r") as text:
            compute_shader_source = text.read()

        # compile the shader
        GL.glShaderSource(self._compute_shader, compute_shader_source)
        GL.glCompileShader(self._compute_shader)

        # get the information from compiling the shader
        status = None
        GL.glGetShaderiv(self._compute_shader, GL.GL_COMPILE_STATUS, status)
        str_info_log: bytes | str = GL.glGetShaderInfoLog(self._compute_shader)
        str_shader_type: str = "compute"
        decoded_info_log: str = (
            str_info_log.decode() if isinstance(str_info_log, bytes) else str_info_log
        )
        print("Compilation result for " + str_shader_type + " shader:\n" + decoded_info_log)

        # link the compute shader with the program
        GL.glAttachShader(self._program_id, self._compute_shader)
        GL.glLinkProgram(self._program_id)
        print("link result", GL.glGetProgramInfoLog(self._program_id))

        self._screen_shader = Shader(
            "./shaders/screen_vertex.glsl", "./shaders/screen_fragment.glsl"
        )

        self._upscale_shader = Shader(
            "./shaders/upscale_vertex.glsl", "./shaders/upscale_fragment.glsl"
        )

        # Create a color texture
        self._color_tex = self._setup_screen_texture(GL.GL_TEXTURE0)
        self._fbo = self._setup_framebuffer(self._color_tex)

    @staticmethod
    def get_instance() -> Renderer:
        """Function for implementing the Singleton behavior."""
        if Renderer.__instance is None:
            Renderer()

        return cast("Renderer", Renderer.__instance)

    def render(self) -> None:
        """Rendering method."""
        # take the starting time before the execution of the rendering function
        start: float = time.time()

        for _i in range(1):
            # if the camera changed, update the flag and reset the accumulation
            if self._camera.changed:
                self._camera.changed = False
                self.reset_accumulation()

            # set the active texture for writing the result of the compute shader to
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
            GL.glBindImageTexture(
                0, self._texture, 0, GL.GL_FALSE, 0, GL.GL_READ_WRITE, GL.GL_RGBA32F
            )

            # use the raytracer shader program to create the frame
            GL.glUseProgram(self._program_id)

            # bind the uniforms required for the rendering of the scene
            GL.glUniformMatrix4fv(
                GL.glGetUniformLocation(self._program_id, "inverse_view_projection"),
                1,
                GL.GL_FALSE,
                self._camera.get_inv_view_proj_matrix(),
            )

            GL.glUniform3f(
                GL.glGetUniformLocation(self._program_id, "eye"),
                self._camera.position[0],
                self._camera.position[1],
                self._camera.position[2],
            )
            GL.glUniform3f(
                GL.glGetUniformLocation(self._program_id, "scene_min"),
                self.scene_min_coords[0],
                self.scene_min_coords[1],
                self.scene_min_coords[2],
            )
            GL.glUniform3f(
                GL.glGetUniformLocation(self._program_id, "scene_extent"),
                self.scene_extent[0],
                self.scene_extent[1],
                self.scene_extent[2],
            )
            GL.glUniform1f(GL.glGetUniformLocation(self._program_id, "time"), float(self._frames))
            GL.glUniform1f(
                GL.glGetUniformLocation(self._program_id, "bounces"), float(self._bounces)
            )
            GL.glUniform3f(
                GL.glGetUniformLocation(self._program_id, "camera_up"),
                self._camera.up[0],
                self._camera.up[1],
                self._camera.up[2],
            )
            GL.glUniform3f(
                GL.glGetUniformLocation(self._program_id, "camera_right"),
                self._camera.right[0],
                self._camera.right[1],
                self._camera.right[2],
            )
            GL.glUniform3f(
                GL.glGetUniformLocation(self._program_id, "camera_front"),
                self._camera.front[0],
                self._camera.front[1],
                self._camera.front[2],
            )

            GL.glViewport(
                0,
                0,
                int(self._width * self._scaling_factor),
                int(self._height * self._scaling_factor),
            )
            # dispatch the compute shader jobs
            GL.glDispatchCompute(
                int((self._width / 8) * self._scaling_factor),
                int((self._height / 4) * self._scaling_factor),
                1,
            )
            GL.glMemoryBarrier(GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

            # clear the buffer
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            # use the screen shader program
            GL.glUseProgram(self._screen_shader.program)
            # bind the required uniforms
            GL.glUniform1i(GL.glGetUniformLocation(self._screen_shader.program, "tex"), 0)
            GL.glUniform1i(GL.glGetUniformLocation(self._screen_shader.program, "old_tex"), 1)
            GL.glUniform1f(
                GL.glGetUniformLocation(self._screen_shader.program, "frames"), self.rendered_frames
            )
            GL.glUniform1f(
                GL.glGetUniformLocation(self._screen_shader.program, "denoise"), self._denoise
            )

            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)

            # bind the screen quad
            GL.glBindVertexArray(self._vao)
            # render to the screen quad
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

            # GL.glUseProgram(self._upscale_shader.program)

            # GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

            # if the renderer is not set to denoise, accumulate the frames
            if self._denoise == 0:
                self.rendered_frames += 1
                GL.glBindTexture(GL.GL_TEXTURE_2D, self._last_frame)
                GL.glCopyTexSubImage2D(
                    GL.GL_TEXTURE_2D,
                    0,
                    0,
                    0,
                    0,
                    0,
                    int(self._width * self._scaling_factor),
                    int(self._height * self._scaling_factor),
                )

            self._frames += 1

            # print()
            # print_progress_bar(i, 199)
            # print(i)

        GL.glUseProgram(self._upscale_shader.program)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self._color_tex)
        GL.glViewport(0, 0, self._width, self._height)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

        # reset to the default VAO
        GL.glBindVertexArray(0)

        end = time.time()
        self.render_time = end - start

    def update_ssbo(self, ssbo: int, elements: NDArray, index: int, unbind: bool = True) -> None:
        """
        Method for updating the content of an SSBO.

        Args:
            ssbo (int): Index of the SSBO object to bind
            elements (NDArray): Array of elements following the STD430 padding convention
            index (int): Index to which bind the SSBO to
            unbind (bool, optional): Whether the buffer should be unbound after. Defaults to True.

        """
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, ssbo)
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, elements.nbytes, elements, GL.GL_STATIC_DRAW)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, index, ssbo)

        if unbind:
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

    def reset_accumulation(self) -> None:
        """Reset the accumulation of the frames."""
        GL.glCopyImageSubData(
            self._texture,
            GL.GL_TEXTURE_2D,
            0,
            0,
            0,
            0,
            self._last_frame,
            GL.GL_TEXTURE_2D,
            0,
            0,
            0,
            0,
            int(self._width * self._scaling_factor),
            int(self._height * self._scaling_factor),
            1,
        )
        self.rendered_frames = 0

    def update_size(self, width: int, height: int) -> None:
        """
        Method for updating the size of the renderer.

        Args:
            width (int): New width
            height (int): New height

        """
        # set the new dimensions
        self._width = is_positive_int(width)
        self._height = is_positive_int(height)

        # update the OpenGL viewport with the new dimensions
        GL.glViewport(0, 0, self._width, self._height)

        # delete the internal textures
        GL.glDeleteTextures(3, [self._texture, self._last_frame, self._color_tex])

        # update the textures with the new dimensions
        self._texture = self._setup_screen_texture(GL.GL_TEXTURE0)
        self._last_frame = self._setup_screen_texture(GL.GL_TEXTURE1)
        self._color_tex = self._setup_screen_texture(GL.GL_TEXTURE0)

        # bind the framebuffer
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        # update the color attachment
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self._color_tex, 0
        )

        # Check framebuffer completeness
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            print("Framebuffer is not complete!")

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    @property
    def bounces(self) -> int:
        """
        Number of bounces each ray takes before getting the result.

        Returns:
            int: Number of bounces

        """
        return self._bounces

    @bounces.setter
    def bounces(self, value: int) -> None:
        self._bounces = is_positive_int(value)

    def _setup_screen_texture(self, texture_slot: Any, scale_filter: Any = GL.GL_NEAREST) -> int:  # noqa: ANN401
        """
        Private method for setting up a screen texture.

        Args:
            texture_slot (Any): GL_TEXTURE<N> to decide the texture slot
            scale_filter (Any): GL_NEAREST / GL_LINEAR

        Returns:
            int: Pointer to the newly created texture

        """
        # create the texture
        texture: int = 0
        texture = GL.glGenTextures(1)
        # set the active texture slot and bind the new texture to it
        GL.glActiveTexture(texture_slot)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        # define the texture parameters
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, scale_filter)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, scale_filter)
        # setup the texture storage size and format
        GL.glTexStorage2D(
            GL.GL_TEXTURE_2D,
            1,
            GL.GL_RGBA32F,
            int(self._width * self._scaling_factor),
            int(self._height * self._scaling_factor),
        )
        GL.glBindImageTexture(0, texture, 1, GL.GL_TRUE, 0, GL.GL_READ_WRITE, GL.GL_RGBA32F)

        # retun the texture
        return texture

    def _setup_framebuffer(self, color_texture: int) -> int:
        """
        Private method for setting up a framebuffer.

        Args:
            color_texture (int): Index to a valid OpenGL texture

        Returns:
            int: OpenGL index of the framebuffer

        """
        # create the framebuffer
        fbo: int = GL.glGenFramebuffers(1)
        # bind it as the current framebuffer
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)

        # attach the input texture to the framebuffer's color attachment
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, color_texture, 0
        )

        # Check framebuffer completeness
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            print("Framebuffer is not complete!")

        # Unbind framebuffer
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        # return the newly created framebuffer
        return fbo


class RendererInitializedError(Exception):
    """Custom exception for when the Window is already initialized."""

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
