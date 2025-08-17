"""Renderer module."""

# ruff: noqa: F403, F405
from __future__ import annotations

import time
from ctypes import *
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from OpenGL import GL

from camera.Camera import Camera
from Shader import Shader

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

    @staticmethod
    def get_instance() -> Renderer:
        """Function for implementing the Singleton behavior."""
        if Renderer.__instance is None:
            Renderer()

        return cast(Renderer, Renderer.__instance)

    def __init__(self) -> None:
        """Init function, it sets up all the requried components for the renderer."""
        # Handle the class as a Singleton
        if Renderer.__instance is not None:
            message: str = "Renderer already exists!"
            raise RendererInitializedError(message)

        # store the instance upon creation
        Renderer.__instance = self

        # setup the dimensions of the renderer
        self.width = 640
        self.height = 480

        # set the clear color for clearing the buffer
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        # enable alpha blending
        GL.glEnable(GL.GL_BLEND)
        # define the blending functions
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        # setup the vertices, VBO and VAO for the screen quad for rendering
        self._setup_screen_quad()

        # setup the screen textures
        self.texture = self._setup_screen_texture(GL.GL_TEXTURE0)
        self.last_frame = self._setup_screen_texture(GL.GL_TEXTURE1)

        # create all the buffers to store all the data to send the shader
        self.vertices = GL.glGenBuffers(1)
        self.model_mats = GL.glGenBuffers(1)
        self.indices = GL.glGenBuffers(1)
        self.normals = GL.glGenBuffers(1)
        self.spheres = GL.glGenBuffers(1)
        self.planes = GL.glGenBuffers(1)
        self.boxes = GL.glGenBuffers(1)
        self.bounding_boxes = GL.glGenBuffers(1)
        self.materials = GL.glGenBuffers(1)
        self.mesh_material_indices = GL.glGenBuffers(1)
        self.bvh = GL.glGenBuffers(1)
        self.triangles = GL.glGenBuffers(1)

        # setup the camera
        self.camera = Camera.getInstance()

        # prepare the render parameters
        self.render_time = 0
        self.rendered_frames = 0
        self.bounces = 3
        self.denoise = 0
        self.far_plane = 1.0

        # create the compute shader and program
        self.compute_shader: int = GL.glCreateShader(GL.GL_COMPUTE_SHADER)
        self.program_id: int = GL.glCreateProgram()

        # read the source of the compute shader
        compute_shader_source: str = ""
        with Path("../shaders/compute.glsl").open("+r") as text:
            compute_shader_source = text.read()

        # compile the shader
        GL.glShaderSource(self.compute_shader, compute_shader_source)
        GL.glCompileShader(self.compute_shader)

        # get the information from compiling the shader
        status = None
        GL.glGetShaderiv(self.compute_shader, GL.GL_COMPILE_STATUS, status)
        str_info_log: bytes | str = GL.glGetShaderInfoLog(self.compute_shader)
        str_shader_type: str = "compute"
        decoded_info_log: str = (
            str_info_log.decode() if isinstance(str_info_log, bytes) else str_info_log
        )
        print("Compilation result for " + str_shader_type + " shader:\n" + decoded_info_log)

        # link the compute shader with the program
        GL.glAttachShader(self.program_id, self.compute_shader)
        GL.glLinkProgram(self.program_id)
        print("link result", GL.glGetProgramInfoLog(self.program_id))

        self.screen_shader = Shader(
            "../shaders/screen_vertex.glsl", "../shaders/screen_fragment.glsl"
        )

    def render(self) -> None:
        """Rendering method."""
        # take the starting time before the execution of the rendering function
        start: float = time.time()

        # if the camera changed, update the flag and reset the accumulation
        if self.camera.changed:
            self.camera.changed = False
            self.reset_accumulation()

        # set the active texture for writing the result of the compute shader to
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
        GL.glBindImageTexture(0, self.texture, 0, GL.GL_FALSE, 0, GL.GL_READ_WRITE, GL.GL_RGBA32F)

        # use the raytracer shader program to create the frame
        GL.glUseProgram(self.program_id)

        # bind the uniforms required for the rendering of the scene
        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self.program_id, "inverse_view_projection"),
            1,
            GL.GL_FALSE,
            self.camera.get_inv_view_proj_matrix(),
        )

        GL.glUniform3f(
            GL.glGetUniformLocation(self.program_id, "eye"),
            self.camera.position[0],
            self.camera.position[1],
            self.camera.position[2],
        )
        GL.glUniform1f(GL.glGetUniformLocation(self.program_id, "time"), float(time.time() % 1))
        GL.glUniform1f(GL.glGetUniformLocation(self.program_id, "bounces"), float(self.bounces))
        GL.glUniform3f(
            GL.glGetUniformLocation(self.program_id, "camera_up"),
            self.camera.up[0],
            self.camera.up[1],
            self.camera.up[2],
        )
        GL.glUniform3f(
            GL.glGetUniformLocation(self.program_id, "camera_right"),
            self.camera.right[0],
            self.camera.right[1],
            self.camera.right[2],
        )
        GL.glUniform3f(
            GL.glGetUniformLocation(self.program_id, "camera_front"),
            self.camera.front[0],
            self.camera.front[1],
            self.camera.front[2],
        )

        # dispatch the compute shader jobs
        GL.glDispatchCompute(int(self.width / 8), int(self.height / 4), 1)
        GL.glMemoryBarrier(GL.GL_ALL_BARRIER_BITS)

        # clear the buffer
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # use the screen shader program
        GL.glUseProgram(self.screen_shader.program)
        # bind the required uniforms
        GL.glUniform1i(GL.glGetUniformLocation(self.screen_shader.program, "tex"), 0)
        GL.glUniform1i(GL.glGetUniformLocation(self.screen_shader.program, "old_tex"), 1)
        GL.glUniform1f(
            GL.glGetUniformLocation(self.screen_shader.program, "frames"), self.rendered_frames
        )
        GL.glUniform1f(GL.glGetUniformLocation(self.screen_shader.program, "denoise"), self.denoise)

        # bind the screen quad
        GL.glBindVertexArray(self.vao)
        # render to the screen quad
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

        # reset to the default VAO
        GL.glBindVertexArray(0)

        # if the renderer is not set to denoise, accumulate the frames
        if self.denoise == 0:
            self.rendered_frames += 1
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.last_frame)
            GL.glCopyTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, 0, 0, self.width, self.height)

        end = time.time()
        self.render_time = end - start

    def update_ssbo(self, ssbo: int, elements: NDArray, index: int, unbind: bool = True) -> None:
        """Method for updating the content of an SSBO.

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
            self.texture,
            GL.GL_TEXTURE_2D,
            0,
            0,
            0,
            0,
            self.last_frame,
            GL.GL_TEXTURE_2D,
            0,
            0,
            0,
            0,
            self.width,
            self.height,
            1,
        )
        self.rendered_frames = 0

    def update_size(self, width: int, height: int) -> None:
        """Method for updating the size of the renderer.

        Args:
            width (int): New width
            height (int): New height

        """
        # set the new dimensions
        self.width = width
        self.height = height

        # update the textures with the new dimensions
        self.texture = self._setup_screen_texture(GL.GL_TEXTURE0)
        self.last_frame = self._setup_screen_texture(GL.GL_TEXTURE1)

    def _setup_screen_quad(self) -> None:
        """Private method for setting up the screeb quad for rendering."""
        # setup the vertices and UVs for the screen quad
        vertices: list[float] = [
            -1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            -1.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            0.0,
            1.0,
            0.0,
        ]

        # convert the vertices into a format for OpenGL
        vertices = (GL.GLfloat * len(vertices))(*vertices)

        # create the VAO and VBO for storing and binding the verrtices of the screen quad
        self.vao: int = 0
        self.vbo: int = 0

        self.vao = GL.glGenVertexArrays(2, self.vao)
        self.vbo = GL.glGenBuffers(1, self.vbo)

        # bind the buffer and store the vertex data
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL.GL_STATIC_DRAW)

        # bind the array and set the 2 entry points for the vertices and the UVs
        GL.glBindVertexArray(self.vao)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(
            0, 3, GL.GL_FLOAT, GL.GL_FALSE, 5 * sizeof(GL.GLfloat), c_void_p(0)
        )
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(
            1, 2, GL.GL_FLOAT, GL.GL_FALSE, 5 * sizeof(GL.GLfloat), c_void_p(3 * sizeof(GL.GLfloat))
        )

    def _setup_screen_texture(self, texture_slot: Any) -> int:  # noqa: ANN401
        """Private method for setting up a screen texture.

        Args:
            texture_slot (Any): GL_TEXTURE<N> to decide the texture slot

        Returns:
            int: Pointer to the newly created texture

        """
        # create the texture
        texture: int = 0
        texture = GL.glGenTextures(1, texture)
        # set the active texture slot and bind the new texture to it
        GL.glActiveTexture(texture_slot)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        # define the texture parameters
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        # setup the texture storage size and format
        GL.glTexStorage2D(GL.GL_TEXTURE_2D, 1, GL.GL_RGBA32F, self.width, self.height)
        GL.glBindImageTexture(0, texture, 1, GL.GL_TRUE, 0, GL.GL_READ_WRITE, GL.GL_RGBA32F)

        # retun the texture
        return texture


class RendererInitializedError(Exception):
    """Custom exception for when the Window is already initialized."""

    def __init__(self, message: str) -> None:
        """Initialization method.

        Args:
            message (str): Message to display in the error

        """
        self.message: str = message
        super().__init__(message)

    def __str__(self) -> str:
        """To string method.

        Returns:
            str: String representation of the object

        """
        return f"{self.message}"
