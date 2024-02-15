import glfw # type: ignore
from OpenGL.GL import * # type: ignore
import OpenGL.GL.shaders # type: ignore


from dataclasses import dataclass
import ctypes
import numpy as np
from texture import Texture
from glfw_window import GLFWWindow
from contextlib import contextmanager
from typing import Iterator, List, Tuple

def noise(w:int, h:int)->np.ndarray:
    return np.random.rand(w,h,4).astype(np.float32)

def assert_image_float_rgba(img:np.ndarray):

    assert len(img.shape) == 3 and img.shape[2] == 4 and img.dtype == np.float32, f"{img.shape}, {img.dtype}"

@contextmanager
def bind_vertex_array(attributeID:int)->Iterator[int]:
    glBindVertexArray(attributeID)
    yield attributeID
    glBindVertexArray(0)

@contextmanager
def bind_buffer(target: GL_ARRAY_BUFFER | GL_ELEMENT_ARRAY_BUFFER, bufferID:int)->Iterator[int]:
    glBindBuffer(target, bufferID)
    yield bufferID
    glBindBuffer(target, 0)

@contextmanager
def bind_texture(target: GL_TEXTURE_2D, textureID:int)->Iterator[int]:
    glBindTexture(target, textureID)
    yield textureID
    glBindTexture(target, 0)

@contextmanager
def use_program(program_id:int)->Iterator[int]:
    glUseProgram(program_id)
    yield program_id
    glUseProgram(0)

@dataclass
class Mesh:
    pos: np.ndarray
    uv: np.ndarray
    color: np.ndarray
    indices: np.ndarray
    image:np.ndarray
    mode: GL_POINTS | GL_LINE_STRIP | GL_LINE_LOOP | GL_LINES | GL_LINE_STRIP_ADJACENCY | GL_LINES_ADJACENCY | GL_TRIANGLE_STRIP | GL_TRIANGLE_FAN | GL_TRIANGLES | GL_TRIANGLE_STRIP_ADJACENCY | GL_TRIANGLES_ADJACENCY | GL_PATCHES=GL_TRIANGLES

    def __post_init__(self):
        assert_image_float_rgba(self.image)
        assert self.pos.dtype == np.float32,    f"'pos' must be np.float32, got: {self.pos.dtype}"
        assert self.uv.dtype == np.float32,     f"'uv' must be np.float32, got: {self.pos.dtype}"
        assert self.indices.dtype == np.uint32, f"'indices' must be np.uint32, got: {self.pos.uint32}"

        # create gl objects
        self.vao = glGenVertexArrays(1)
        self.pos_vbo = glGenBuffers(1)
        self.uv_vbo  =glGenBuffers(1)
        self.color_vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)
        self.tex = glGenTextures(1)
        self.program = None

        # create shader
        vertex_shader = """
        #version 330
        in layout(location = 0) vec3 position;
        in layout(location = 1) vec2 uv;
        in layout(location = 2) vec4 color;
        
        out vec4 vColor;
        out vec2 outTexCoords;
        void main()
        {
            gl_Position = vec4(position, 1.0f);
            vColor = color;
            outTexCoords = uv;
        }
        """

        fragment_shader = """
        #version 330
        in vec2 outTexCoords;
        in vec4 vColor;
        out vec4 outColor;
        uniform sampler2D samplerTex;
        void main()
        {
            vec4 texel = texture(samplerTex, outTexCoords);
            outColor = texel * vColor;
        }
        """
        vertex_shader = OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
        fragment_shader = OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        self.program = OpenGL.GL.shaders.compileProgram(vertex_shader, fragment_shader)
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

        # create geometry
        # #          pos              color          uv
        # vertex = [(-0.5, -0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        #           ( 0.5, -0.5, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0),
        #           ( 0.5,  0.5, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        #           (-0.5,  0.5, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0)]

        # begin vertex attributes
        with bind_vertex_array(self.vao):
            loc = glGetAttribLocation(self.program, "position")
            with bind_buffer(GL_ARRAY_BUFFER, self.pos_vbo):
                glBufferData(GL_ARRAY_BUFFER, self.pos.nbytes, self.pos, GL_DYNAMIC_DRAW)
                glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
                glEnableVertexAttribArray(loc)

            loc = glGetAttribLocation(self.program, "uv")
            with bind_buffer(GL_ARRAY_BUFFER, self.uv_vbo):
                glBufferData(GL_ARRAY_BUFFER, self.uv.nbytes, self.uv, GL_DYNAMIC_DRAW)
                glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
                glEnableVertexAttribArray(loc)

            loc = glGetAttribLocation(self.program, "color")
            with bind_buffer(GL_ARRAY_BUFFER, self.color_vbo):
                glBufferData(GL_ARRAY_BUFFER, self.color.nbytes, self.color, GL_DYNAMIC_DRAW)
                glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
                glEnableVertexAttribArray(loc)

        # vertex indices
        with bind_buffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo):
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_DYNAMIC_DRAW)

        # create texture
        with bind_texture(GL_TEXTURE_2D, self.tex):
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT) # texture wrapping params
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR) # texture filtering params
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            # glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            h, w, c = self.image.shape
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_FLOAT, self.image)

    def __del__(self):
        """__del__ might be called after glfw terminate when python is shutting dowwn.
        At this point the context is destroyed, and glfw is probably already Uninitalized. 
        I dont like ghaving these errors everz time I close the app, so e are ignoring errors on deletion as a workaround for now.
        TODO: this shoud be fixed, to make sure, error on deletion are not ignored. Maybe keep an _initalized_ boolean flag on the window class, and check that
        """

        try: glDeleteBuffers(1, [self.pos_vbo])
        except GLError as err: pass

        try: glDeleteBuffers(1, [self.uv_vbo])
        except GLError as err: pass

        try: glDeleteBuffers(1, [self.color_vbo])
        except GLError as err: pass

        try: glDeleteBuffers(1, [self.ebo])
        except GLError as err: pass

        try: glDeleteVertexArrays(1, [self.vao])
        except GLError as err: pass

        try: glDeleteTextures(1, [self.tex])
        except GLError as err: pass

        try: glDeleteProgram(self.program)
        except GLError as err: pass

    @classmethod
    def imagePlane(klass, image:np.ndarray):
        mesh = Mesh(pos=np.array([    (-0.5, -0.5, 0.0),
                                      ( 0.5, -0.5, 0.0),
                                      ( 0.5,  0.5, 0.0),
                                      (-0.5,  0.5, 0.0)], dtype=np.float32),
                    uv=np.array([     (0.0, 0.0),
                                      (1.0, 0.0),
                                      (1.0, 1.0),
                                      (0.0, 1.0)],        dtype=np.float32),
                    color=np.array([  (1.0, 0.0, 0.0, 1.0),
                                      (0.0, 1.0, 0.0, 1.0),
                                      (0.0, 0.0, 1.0, 1.0),
                                      (1.0, 1.0, 1.0, 1.0)],   dtype=np.float32),
                    indices=np.array([(0,1,2), (0,2,3)],  dtype=np.uint32),
                    image=image,
                    mode=GL_TRIANGLES)
        return mesh

    @classmethod
    def points(klass, coords:np.ndarray | List[Tuple[float, float, float]]):
        n = len(coords)
        mesh = Mesh(pos=    np.array(coords, dtype=np.float32),
                    uv=     np.linspace(0,1,99, dtype=np.float32),
                    color=  np.ones(shape=(n, 4), dtype=np.float32),
                    indices=np.arange(0,n,1, dtype=np.uint32),
                    image=  np.ones(shape=(2,2,4), dtype=np.float32),
                    mode=GL_POINTS)
        return mesh

    @classmethod
    def rect(klass, x,y,w,h):
        raise NotImplementedError
        return mesh

    def draw(self):
        with use_program(self.program):
            with bind_vertex_array(self.vao):
                with bind_texture(GL_TEXTURE_2D, self.tex):
                    with bind_buffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo):
                        glDrawElements(self.mode, self.indices.size, GL_UNSIGNED_INT, None)


class MyWindow(GLFWWindow):
    def __init__(self):
        super().__init__()
        
        
    def draw(self):
        super().draw()
        self.imageplane = Mesh.imagePlane(image=noise(128, 126))
        self.imageplane.draw()

        coords = np.array([(-0.5, -0.5, 0.0),
              ( 0.5, -0.5, 0.0),
              ( 0.5,  0.5, 0.0),
              (-0.5,  0.5, 0.0)], dtype=np.float32)
        coords+=(np.random.rand(*coords.shape)-np.array([0.5, 0.5, 0.5]))*0.1
        self.points = Mesh.points(coords=coords)
        self.points.draw()
        # self.imageplane = ImagePlane(noise(128, 126))
        # self.imageplane.draw()
        # self.imageplane2 = ImagePlane(noise(128, 126))
        # self.imageplane2.draw()
        # self.points = Points(
        # self.points.draw()
        # img = np.random.rand(100,100,4)
        # img = (img * 255).astype(np.uint8)
        # tex = Texture(img)

        # glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

    def on_cursor_pos(self, window, x, y):
        print(x, y)

if __name__ == "__main__":
    app = MyWindow()
    app.start()
