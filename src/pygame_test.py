import pygame
from OpenGL.GL import *
from ctypes import sizeof, c_void_p

pygame.init()
display = pygame.display.set_mode((800, 600), pygame.DOUBLEBUF|pygame.OPENGL)
clock = pygame.time.Clock()
FPS = 60


VERTEX_SHADER_SOURCE = '''
    #version 330 core
    layout (location = 0) in vec3 aPos;
    
    void main()
    {
        gl_Position = vec4(aPos, 1.0);
    }
'''

FRAGMENT_SHADER_SOURCE = '''
    #version 330 core

    out vec4 fragColor;
    void main()
    {
        fragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
    }
'''

#  (x, y, z)
vertices = [
    -0.5, -0.5, 0.0,
    0.5, -0.5, 0.0,
    0.0, 0.5, 0.0,
]
vertices = (GLfloat * len(vertices))(*vertices)

# we created program object 
program = glCreateProgram()

# we created vertex shader
vertex_shader = glCreateShader(GL_VERTEX_SHADER)
# we passed vertex shader's source to vertex_shader object
glShaderSource(vertex_shader, VERTEX_SHADER_SOURCE)
# and we compile it
glCompileShader(vertex_shader)


# we created fragment shader
fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
# we passed fragment shader's source to fragment_shader object
glShaderSource(fragment_shader, FRAGMENT_SHADER_SOURCE)
# and we compile it
glCompileShader(fragment_shader)

# attach these shaders to program
glAttachShader(program, vertex_shader)
glAttachShader(program, fragment_shader)

# link the program
glLinkProgram(program)

# create vbo object
vbo = None
vbo = glGenBuffers(1, vbo)

# enable buffer(VBO)
glBindBuffer(GL_ARRAY_BUFFER, vbo)

# send the data  
glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW)

# create vao object
vao = None
vao = glGenVertexArrays(1, vao)

# enable VAO and then finally binding to VBO object what we created before.
glBindVertexArray(vao)

# we activated to the slot of position in VAO (vertex array object)
glEnableVertexAttribArray(0)

# explaining to the VAO what data will be used for slot 0 (position slot) 
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), c_void_p(0))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    glClearColor(1.0, 0.6, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)

    glUseProgram(program)
    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLES, 0, 3)
    
    pygame.display.flip()
    clock.tick(FPS)