import numpy as np
from OpenGL.GL import *

class Texture:
    def __init__(self, data:np.ndarray):
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        # texture wrapping params
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        # texture filtering params
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)


        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        h, w, c = data.shape
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
        self.texID = tex
        # print(f"create texture: {self.texID}, in context: {ctx}")

    @classmethod
    def from_data(kls, data:np.ndarray):
        return Texture(data=data)

    def __copy__(self):
        raise NotImplementedError

    def __del__(self):
        glDeleteTextures(1,[self.texID])
        # sys.getrefcount(self) <=3: # 3 is the magic number: 1st ref is the object, 2nd is 