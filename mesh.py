from typing import Iterator, List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from OpenGL.GL import * # type: ignore
import OpenGL.GL.shaders # type: ignore

import glm
def assert_image_float_rgba(img:NDArray[np.float32]):
    assert img.ndim == 3 and img.shape[2]== 4 and img.dtype == np.float32, f"{img.shape}, {img.dtype}"

from bind_with_context import *

from numpy.typing import NDArray

class Mesh:
    def __init__(self, 
               positions:NDArray[np.float32] | List[Tuple[float, float, float]], 
               uvs:NDArray[np.float32] | List[Tuple[float, float]], 
               colors:NDArray[np.float32] | List[Tuple[float, float, float, float]], 
               indices:NDArray[np.uint32] | List[int],
               image:NDArray[np.float32],
               projection:glm.mat4=glm.ortho(-1,1,-1,1,-1,1),
               view:glm.mat4=glm.mat4(),
               mode=GL_TRIANGLES):

        # validate input
        assert_image_float_rgba(image)
        assert positions.dtype == np.float32,    f"'pos' must be np.float32, got: {positions.dtype}"
        assert uvs.dtype == np.float32,     f"'uv' must be np.float32, got: {uvs.dtype}"
        assert colors.dtype == np.float32,     f"'uv' must be np.float32, got: {colors.dtype}"
        assert indices.dtype == np.uint32, f"'indices' must be np.uint32, got: {indices.dtype}"

        # check consistency
        assert len(positions) == len(uvs) == len(colors), f"all attributes must have the same length, got: {len(positions), len(uvs), len(colors), len(indices)}"

        # consolidate inputs
        self.positions = positions
        self.uvs = uvs
        self.colors = colors
        self.indices = indices
        self.image = image
        self.projection = projection
        self.view = view
        self.mode = mode

        # SETUP GL
        # create shader program
        vertex_shader = """
        #version 330
        uniform mat4 viewMatrix;
        uniform mat4 projectionMatrix;
        in layout(location = 0) vec3 position;
        in layout(location = 1) vec2 uv;
        in layout(location = 2) vec4 color;
        
        out vec4 vColor;
        out vec2 outTexCoords;
        void main()
        {
            gl_Position = projectionMatrix * viewMatrix * vec4(position, 1.0f);
            gl_PointSize = 4.0;
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

        self.vao = glGenVertexArrays(1)
        with bind_vertex_array(self.vao):
            loc = glGetAttribLocation(self.program, "position")
            self.posvbo = glGenBuffers(1)
            with bind_buffer(GL_ARRAY_BUFFER, self.posvbo):
                glBufferData(GL_ARRAY_BUFFER, self.positions.nbytes, self.positions, GL_DYNAMIC_DRAW)
                glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
                glEnableVertexAttribArray(loc)

            loc = glGetAttribLocation(self.program, "uv")
            self.uvvbo  =glGenBuffers(1)
            with bind_buffer(GL_ARRAY_BUFFER, self.uvvbo):
                glBufferData(GL_ARRAY_BUFFER, self.uvs.nbytes, self.uvs, GL_DYNAMIC_DRAW)
                glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
                glEnableVertexAttribArray(loc)

            loc = glGetAttribLocation(self.program, "color")
            self.colorvbo = glGenBuffers(1)
            with bind_buffer(GL_ARRAY_BUFFER, self.colorvbo):
                glBufferData(GL_ARRAY_BUFFER, self.colors.nbytes, self.colors, GL_DYNAMIC_DRAW)
                glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
                glEnableVertexAttribArray(loc)

        self.ebo = glGenBuffers(1)
        with bind_buffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo):
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_DYNAMIC_DRAW)

        # create texture
        self.tex = glGenTextures(1)
        with bind_texture(GL_TEXTURE_2D, self.tex):
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT) # texture wrapping params
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR) # texture filtering params
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            # glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            h, w, c = self.image.shape
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_FLOAT, self.image)


    def __del__(self):
        """__del__ might be called after glfw terminate when python is shutting dowwn.
        At this point the context is destroyed, and glfw is probably already Uninitalized. 
        I dont like ghaving these errors everz time I close the app, so e are ignoring errors on deletion as a workaround for now.
        TODO: this shoud be fixed, to make sure, error on deletion are not ignored. Maybe keep an _initalized_ boolean flag on the window class, and check that
        """

        try: glDeleteBuffers(1, [self.posvbo])
        except GLError as err: pass

        try: glDeleteBuffers(1, [self.uvvbo])
        except GLError as err: pass

        try: glDeleteBuffers(1, [self.colorvbo])
        except GLError as err: pass

        try: glDeleteBuffers(1, [self.ebo])
        except GLError as err: pass

        try: glDeleteVertexArrays(1, [self.vao])
        except GLError as err: pass

        try: glDeleteTextures(1, [self.tex])
        except GLError as err: pass

        try: glDeleteProgram(self.program)
        except GLError as err: pass

    def update(self, 
               positions:Optional[NDArray[np.float32] | List[Tuple[float, float, float]]]=None, 
               uvs:Optional[NDArray[np.float32] | List[Tuple[float, float]]]=None, 
               colors:Optional[NDArray[np.float32] | List[Tuple[float, float, float, float]]]=None, 
               indices:Optional[NDArray[np.uint32] | List[int]]=None,
               image:Optional[NDArray[np.float32]]=None,
               projection:Optional[glm.mat4]=None,
               view:Optional[glm.mat4]=None):
        """update mesh attributes"""
        positions = np.array(positions, dtype=np.float32) if positions is not None else self.positions
        uvs =       np.array(uvs,       dtype=np.float32) if uvs is not None       else self.uvs
        colors =    np.array(colors,    dtype=np.float32) if colors is not None    else self.colors
        indices =   np.array(indices,   dtype=np.uint32)  if indices is not None   else self.indices

        # check consistency
        assert len(positions) == len(uvs) == len(colors), f"all attributes must have the same length, got: {len(self.positions), len(self.uvs), len(self.colors), len(self.indices)}"

        # update GPU
        with bind_vertex_array(self.vao):
            if positions is not self.positions:
                self.positions = positions
                glDeleteBuffers(1, [self.posvbo])
                self.posvbo = glGenBuffers(1)
                loc = glGetAttribLocation(self.program, "position")
                with bind_buffer(GL_ARRAY_BUFFER, self.posvbo):
                    glBufferData(GL_ARRAY_BUFFER, self.positions.nbytes, self.positions, GL_DYNAMIC_DRAW)
                    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
                    glEnableVertexAttribArray(loc)
            if uvs is not self.uvs:
                self.uvs = self.uvs
                glDeleteBuffers(1, [self.uvvbo])
                self.uvvbo = glGenBuffers(1)
                loc = glGetAttribLocation(self.program, "uv")
                with bind_buffer(GL_ARRAY_BUFFER, self.uvvbo):
                    glBufferData(GL_ARRAY_BUFFER, self.uvs.nbytes, self.uvs, GL_DYNAMIC_DRAW)
                    glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
                    glEnableVertexAttribArray(loc)

            if colors is not self.colors:
                self.colors = colors
                glDeleteBuffers(1, [self.colorvbo])
                self.colorvbo = glGenBuffers(1)
                loc = glGetAttribLocation(self.program, "color")
                with bind_buffer(GL_ARRAY_BUFFER, self.colorvbo):
                    glBufferData(GL_ARRAY_BUFFER, self.colors.nbytes, self.colors, GL_DYNAMIC_DRAW)
                    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
                    glEnableVertexAttribArray(loc)

        # vertex indices
        if indices is not self.indices:
            self.indices = indices
            glDeleteBuffers(1, [self.ebo])
            self.ebo = glGenBuffers(1)

            with bind_buffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo):
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_DYNAMIC_DRAW)

        image = image if image is not None else self.image
        if image is not self.image:
            self.image = image
            glDeleteTextures(1, [self.tex])
            self.tex = glGenTextures(1)
            with bind_texture(GL_TEXTURE_2D, self.tex):
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT) # texture wrapping params
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR) # texture filtering params
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                # glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
                glPixelStorei(GL_PACK_ALIGNMENT, 4)
                h, w, c = self.image.shape
                assert c==4
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_FLOAT, self.image)

        if projection is not None:
            self.projection = projection
        if view is not None:
            self.view = view

    def render(self):
        # Get the uniform locations
        with use_program(self.program):
            projection_location = glGetUniformLocation(self.program, "projectionMatrix")
            view_location = glGetUniformLocation(self.program, "viewMatrix")
            glUniformMatrix4fv(projection_location, 1, GL_FALSE, glm.value_ptr(self.projection))
            glUniformMatrix4fv(view_location, 1, GL_FALSE, glm.value_ptr(self.view))
            with bind_vertex_array(self.vao):
                with bind_texture(GL_TEXTURE_2D, self.tex):
                    with bind_buffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo):
                        glDrawElements(self.mode, self.indices.size, GL_UNSIGNED_INT, None)

class ImagePlane(Mesh):
    def __init__(self, image:np.ndarray):
        super().__init__(
            positions=np.array([    (-0.5, -0.5, 0.0),
                              ( 0.5, -0.5, 0.0),
                              ( 0.5,  0.5, 0.0),
                              (-0.5,  0.5, 0.0)], dtype=np.float32),
            uvs=np.array([     (0.0, 0.0),
                              (1.0, 0.0),
                              (1.0, 1.0),
                              (0.0, 1.0)],        dtype=np.float32),
            colors=np.array([ (1.0, 1.0, 1.0, 1.0),
                              (1.0, 1.0, 1.0, 1.0),
                              (1.0, 1.0, 1.0, 1.0),
                              (1.0, 1.0, 1.0, 1.0)],   dtype=np.float32),
            indices=np.array([(0,1,2), (0,2,3)],  dtype=np.uint32),
            image=image,
            mode=GL_TRIANGLES)


class PointCloud(Mesh):
    def __init__(self, positions=np.ndarray, colors:Optional[np.ndarray]=None):
        super().__init__(
            positions=np.array(positions, dtype=np.float32),
            uvs = np.zeros((positions.shape[0], 2), dtype=np.float32),  # Placeholder UVs
            colors = np.array(colors, dtype=np.float32) if colors is not None else np.ones(shape=(positions.shape[0], 4), dtype=np.float32),
            indices = np.arange(0, positions.shape[0], dtype=np.uint32),
            image = np.ones(shape=(2, 2, 4), dtype=np.float32),  # Placeholder image
            mode=GL_POINTS)


class Rectangle(Mesh):
    def __init__(self, rect, color=None):
        x,y,w,h = rect
        positions = np.array([
            (x    , y    , 0.0),
            (x + w, y    , 0.0),
            (x + w, y + h, 0.0),
            (x    , y + h, 0.0)
        ], dtype=np.float32)

        uvs = np.array([
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0)
        ], dtype=np.float32)

        if color is not None:
            colors = np.ones(shape=(positions.shape[0], 4), dtype=np.float32)
            colors[:] = color
        else:
            colors = np.ones(shape=(positions.shape[0], 4), dtype=np.float32)

        indices = np.array([
            (0, 1, 2),
            (0, 2, 3)
        ], dtype=np.uint32)

        image = np.zeros(shape=(2, 2, 4), dtype=np.float32)  # Placeholder image

        super().__init__(
            positions=positions,
            uvs = uvs,
            colors = colors,
            indices = indices,
            image = np.ones(shape=(2, 2, 4), dtype=np.float32),  # Placeholder image
            mode=GL_TRIANGLES)

    def update(self, rect:Optional[Tuple[float, float, float, float]]=None, projection=None, view=None):
        if rect is not None:
            x,y,w,h = rect
            positions = np.array([
                (x    , y    , 0.0),
                (x + w, y    , 0.0),
                (x + w, y + h, 0.0),
                (x    , y + h, 0.0)
            ], dtype=np.float32)
        else:
            positions = None
        super().update(positions=positions, view=view, projection=projection)
