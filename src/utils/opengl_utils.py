"""OpenGL Utilities."""

from ctypes import c_void_p, sizeof

from OpenGL import GL


def create_screen_quad_vao() -> int:
    """Method for setting up the screeb quad for rendering."""
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
    vao: int = 0
    vbo: int = 0

    vao = GL.glGenVertexArrays(2, vao)
    vbo = GL.glGenBuffers(1, vbo)

    # bind the buffer and store the vertex data
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL.GL_STATIC_DRAW)

    # bind the array and set the 2 entry points for the vertices and the UVs
    GL.glBindVertexArray(vao)
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 5 * sizeof(GL.GLfloat), c_void_p(0))
    GL.glEnableVertexAttribArray(1)
    GL.glVertexAttribPointer(
        1, 2, GL.GL_FLOAT, GL.GL_FALSE, 5 * sizeof(GL.GLfloat), c_void_p(3 * sizeof(GL.GLfloat))
    )

    return vao
