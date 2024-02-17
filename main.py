import glfw # type: ignore
from OpenGL.GL import * # type: ignore
import OpenGL.GL.shaders # type: ignore
from typing import Callable, Any

from dataclasses import dataclass, field
import ctypes
import numpy as np
from texture import Texture
from glfw_window import GLFWWindow
from contextlib import contextmanager
from typing import Iterator, List, Tuple
import glm


from mesh import Mesh, ImagePlane, PointCloud, Rectangle
import time
from mousetools import SelectionTool, MoveTool, MouseState


type Point = Tuple[float, float] #x,y
type Rect = Tuple[float, float, float, float] # x,y,w,h

def noise(w:int=256, h:int=256)->np.ndarray:
    return np.random.rand(h,w,4).astype(np.float32)

def constant(w:int=256, h:int=256, color=(1,1,1,1)):
    img = np.empty(shape=(h,w,4))
    img[:] = color
    return img

def checker(w:int=256, h:int=256, tile_size:Tuple[int,int]=32, colorA=(0,0,0,1), colorB=(1,1,1,1)):
    indices = np.indices((h, w, 4))
    return np.where(( (indices[0]+w/2) // tile_size[0] + (indices[1]+h/2) // tile_size[1]) % 2 == 0, colorA, colorB).astype(np.float32)


from skimage.transform import ProjectiveTransform
def cornerpin(im_src: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    height, width, dim = im_src.shape
    src, dst = src_pts[:, [1, 0]], dst_pts[:, [1, 0]]
    pt = ProjectiveTransform()
    pt.estimate(src, dst)
    x, y = np.mgrid[:height, :width]
    dst_indices = np.hstack((x.reshape(-1, 1), y.reshape(-1,1))) 
    src_indices = np.round(pt.inverse(dst_indices), 0).astype(int)
    valid_idx = np.where((src_indices[:,0] < height) & (src_indices[:,1] < width) & (src_indices[:,0] >= 0) & (src_indices[:,1] >= 0))
    dst_indicies_valid = dst_indices[valid_idx]
    src_indicies_valid = src_indices[valid_idx]

    im_dst = np.zeros(shape=im_src.shape)
    im_dst[dst_indicies_valid[:,0],dst_indicies_valid[:,1]] = im_src[src_indicies_valid[:,0],src_indicies_valid[:,1]]
    return im_dst.astype(np.float32) 

def project(points:np.ndarray, viewport:Rect, projection:glm.mat4=glm.mat4(), view:glm.mat4=glm.mat4(), model:glm.mat4=glm.mat4(), flip_vertical=True):
    projected_points = []
    for point in points:
        x, y, z = glm.project(glm.vec3(point[0], -point[1] if flip_vertical else point[1], point[2]), view*model, projection, glm.vec4(*viewport))
        projected_points.append((x, y, z))
    return np.array(projected_points)

def unproject(points:np.ndarray, viewport:Rect, projection:glm.mat4=glm.mat4(), view:glm.mat4=glm.mat4(), model:glm.mat4=glm.mat4(), flip_vertical=True):
    unprojected_points = []
    for point in points:
        x, y, z = glm.unProject(glm.vec3(point[0], point[1], point[2]), view*model, projection, glm.vec4(*viewport))
        unprojected_points.append((x, -y if flip_vertical else y, z))
    return np.array(unprojected_points)

class MyWindow(GLFWWindow):
    def __init__(self):
        super().__init__()
        # Create PointCloud
        self.points = PointCloud(
            positions=np.array([(-0.5, -0.5, 0),
                                ( 0.5, -0.5, 0),
                                ( 0.5,  0.5, 0),
                                (-0.5,  0.5, 0)], dtype=np.float32),
            colors=np.array([(1,0,0,1),
                             (0,1,0,1),
                             (0.3,.3,1,1),
                             (1,1,0,1)], dtype=np.float32))

        # Create selection Rectangle
        self.bbox = Rectangle((0, 0, 0.5, 0.5), color=(.4,.6,1,.2))

        # Create ImagePlane
        self.imageplane = ImagePlane(image=noise(128, 128))

        # Create MouseTools
        self.selectiontool = SelectionTool(candidates=unproject(self.points.positions, (0,0,self.width, self.height)))
        self.movetool = MoveTool()

        self.on_resize(self.window, self.width, self.height)

    def on_resize(self, window, w, h):
        super().on_resize(window, w, h)
        # camera projection
        aspect = self.width/self.height
        self.projection = glm.ortho(-aspect,aspect,-1, 1,-100,100)
        # projection = glm.perspective(45.0, aspect, 1.0, 150.0);
        self.view = glm.translate((0,0,1))

    def render(self):
        super().render()

        # Switch tools
        if glfw.get_key(self.window, glfw.KEY_B) == glfw.PRESS:
            # TODO: activate selection tool
            print("b is down")

        if glfw.get_key(self.window, glfw.KEY_G) == glfw.PRESS:
            # TODO: actiavet move tool
            selection = self.selectiontool.selection
            move = self.selectiontool.mousedelta
            move=(move[0]/self.width*2, -move[1]/self.height*2, 0)
            new_positions = self.points.positions
            new_positions[selection]+=move
            self.points.update(positions=new_positions)

        # Animate Points
        # wiggle points
        self.points.update(colors=np.random.rand(*self.points.colors.shape)*0.3+0.5)
        self.points.update(positions=self.points.positions+np.random.rand(*self.points.positions.shape)*0.001-0.001/2)

        # set points color on selection
        self.selectiontool.candidates = project(self.points.positions, viewport=(0,0,self.width, self.height), projection=self.projection, view=self.view)
        self.selectiontool.update(mousepos=glfw.get_cursor_pos(self.window), leftbutton=glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        
        colors = self.points.colors
        colors[self.selectiontool.selection] = (1,1,1,1)
        self.points.update(colors = colors)

        # Selection Rectangle
        if self.selectiontool.state == MouseState.DRAG:
            x, y = self.selectiontool.mousepos
            x2, y2 = self.selectiontool.mousebegin
            projected = unproject(np.array([(x,y, 0), (x2, y2, 0)]), 
                viewport=(0,0,self.width, self.height), 
                projection=self.projection,
                view=self.view)
        
            x,y = projected[0][0], projected[0][1]
            x2, y2 = projected[1][0], projected[1][1]
            w, h = x2-x, y2-y

            self.bbox.update(rect=(x,y,w, h))
        elif self.selectiontool.state == MouseState.IDLE:
            x, y = self.selectiontool.mousepos
            projected = unproject(np.array([(x,y, 0)]), 
                viewport=(0,0,self.width, self.height), 
                projection=self.projection,
                view=self.view)
        
            w, h = 0.01, 0.01
            x,y = projected[0][0]-w/2, projected[0][1]-h/2
            self.bbox.update(rect=(x,y,w, h))

        # Imageplane
        # img = constant(256, 256, color=(1,1,1,1))
        img = checker(256, 256, (32,32))
        h,w,c = img.shape
        model = glm.scale((2,2,2))
        dst = project(self.points.positions, viewport=(0,0,w, h), view=glm.mat4(), model=model, projection=glm.ortho(-1,1,-1,1,-1,1), flip_vertical=False).astype(np.float32)
        img = cornerpin(im_src=img, 
                        src_pts=np.array([(0, 0),
                                          (w, 0),
                                          (w, h),
                                          (0, h)], dtype=np.float32),
                        dst_pts= dst)
        
        self.imageplane.update(image=img, projection=self.projection, view=self.view)

        # Render objects
        renderables = [self.imageplane, self.points, self.bbox]

        # update projection
        for obj in renderables:
            obj.update(projection=self.projection, view=self.view)

        # render
        for obj in renderables:
            obj.render()


    def on_cursor_pos(self, window, x, y):
        pass




if __name__ == "__main__":
    app = MyWindow()
    app.start()

