# opengl
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders

# 
import numpy as np
from texture import Texture
from glfw_window import GLFWWindow

import cv2

class MyWindow(GLFWWindow):
    def __init__(self):
        super().__init__()
        """INITALIZE YOUT STUFF HERE"""
        print("create geometry")
        quad = [-0.5, -0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                0.5, -0.5, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                -0.5, 0.5, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]

        quad = np.array(quad, dtype=np.float32)

        indices = [0, 1, 2,
                   2, 3, 0]

        indices = np.array(indices, dtype=np.uint32)

        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, 128, quad, GL_STATIC_DRAW)

        EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 24, indices, GL_STATIC_DRAW)

        # position = glGetAttribLocation(shader, "position")
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # color = glGetAttribLocation(shader, "color")
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        # texCoords = glGetAttribLocation(shader, "inTexCoords")
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24))
        glEnableVertexAttribArray(2)

        print("create texture")
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        # texture wrapping params
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        # texture filtering params
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        img = np.random.rand(100,100,4)
        img = (img * 255).astype(np.uint8)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        h, w, c = img.shape
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img)

        print("create shader")
        vertex_shader = """
        #version 330
        in layout(location = 0) vec3 position;
        in layout(location = 1) vec3 color;
        in layout(location = 2) vec2 inTexCoords;
        out vec3 newColor;
        out vec2 outTexCoords;
        void main()
        {
            gl_Position = vec4(position, 1.0f);
            newColor = color;
            outTexCoords = inTexCoords;
        }
        """

        fragment_shader = """
        #version 330
        in vec3 newColor;
        in vec2 outTexCoords;
        out vec4 outColor;
        uniform sampler2D samplerTex;
        void main()
        {
            outColor = texture(samplerTex, outTexCoords);
        }
        """
        shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                                  OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

        glUseProgram(shader)

    def draw(self):
        super().draw()


        img = np.random.rand(100,100,4)
        src_points = np.array([
            (0.0, 1.0), (1.0, 1.0),
            (0.0, 0.0), (0.0, 1.0)
        ], dtype=np.float32)
        dst_points = np.array([
            (0.0, 1.0), (1.0, 1.0),
            (0.0, 0.0), (0.0, 1.1)
        ], dtype=np.float32)
        m = cv2.getPerspectiveTransform(src_points, dst_points)
        img = cv2.perspectiveTransform(img, m)

        print(res)

        img_uint8 = (img * 255).astype(np.uint8)
        tex = Texture(img_uint8)

        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

    def on_cursor_pos(self, window, x, y):
        print(x, y)

if __name__ == "__main__":
    app = MyWindow()
    app.start()

