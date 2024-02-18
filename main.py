import glfw # type: ignore
from OpenGL.GL import * # type: ignore
import OpenGL.GL.shaders # type: ignore
from typing import Callable, Any

from dataclasses import dataclass, field
import ctypes
import numpy as np
from texture import Texture
from glfw_window import Window
from contextlib import contextmanager
from typing import Iterator, List, Tuple
import glm


import mesh
import time
from mousetools import SelectionTool, MoveTool

import math
from typing import Tuple
import math

from functools import cache

def subtract(V1, V2):
    return V1[0]-V2[0], V1[1]-V2[1]

def length(V):
    return math.sqrt(V[0] ** 2 + V[1] ** 2)

def distance(P1, P2):
    V = subtract(P1,P2)
    return length(V)

def rect_contains(rect:Tuple[float, float, float, float], P:Tuple[float, float]):
    x,y,w,h = rect
    return P[0]>x and P[0]<x+w and P[1]>y and P[1]<y+h

def rect_from_corners(x1,y1,x2,y2):
    x = min(x1,x2)
    y = min(y1,y2)
    w = max(x1,x2)-x
    h = max(y1,y2)-y
    return x,y,w,h

type Point = Tuple[float, float] #x,y
type Rect = Tuple[float, float, float, float] # x,y,w,h

def noise(w:int=256, h:int=256)->np.ndarray:
    return np.random.rand(h,w,4).astype(np.float32)

@cache
def constant(w:int=256, h:int=256, color=(1,1,1,1)):
    return np.full(shape=(h,w,4), fill_value=color, dtype=np.float32)
    img = np.empty(shape=(h,w,4))
    img[:] = color
    return img

@cache
def checker(w:int=256, h:int=256, tile_size:Tuple[int,int]=(32, 32), colorA=(0,0,0,1), colorB=(1,1,1,1)):
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


@dataclass
class ViewModel:
    controlpoints: np.ndarray
    image: np.ndarray
    selection: List
    # def __init__(self):
    #     self.src_corners = np.array([(-0.5, -0.5, 0),
    #                         ( 0.5, -0.5, 0),
    #                         ( 0.5,  0.5, 0),
    #                         (-0.5,  0.5, 0)], dtype=np.float32)
    #     self.dst_corners = np.copy(self.src_corners)
    #     self.image = checker(256,256, (32,32))
    #     self.selection = []
        
    def move(self, x,y, z=0):
        self.controlpoints[self.selection]+=x,y,z

    def select(self, pos:Tuple[float, float]):
        threshold = 0.07
        x1 = pos[0]-threshold
        x2 = pos[0]+threshold
        y1 = pos[1]-threshold
        y2 = pos[1]+threshold

        self.selection = []
        for i, (x,y,z) in enumerate(self.controlpoints):
            if x1<x and x<x2 and y1<y and y<y2:
                self.selection = [i]
                break

    def perform_box_selection(self, rect:Rect):
        """
        rect in world space
        """
        selection = []
        x1,y1,w,h = rect
        x2,y2 = x1+w, y1+h
        for i, (x,y,z) in enumerate(self.controlpoints):
            if x1<x and x<x2 and y1<y and y<y2:
                selection.append(i)

        self.selection = selection

    @property
    def src_corners(self):
        return self.controlpoints[:4]

    @property
    def dst_corners(self):
        return self.controlpoints[4:8]
    
        

if __name__ == "__main__":
    model = ViewModel(controlpoints = np.array([
        (-0.5, -0.5,0.0), ( 0.5, -0.5,0.0), ( 0.5,  0.5,0.0), (-0.5,  0.5,0.0)], dtype=np.float32),
                  image = checker(256,256, (32,32)),
                  selection = [])
    window = Window()

    pointcloud = mesh.PointCloud(positions=model.controlpoints, colors=np.full(shape=(model.controlpoints.shape[0],4), fill_value=(0.5, 0.5, 0.5, 1.0), dtype=np.float32))
    imageplane = mesh.ImagePlane(image=model.image)
    selection_rect = mesh.Rectangle((0, 0, 0.0, 0.0), color=(.4,.6,1,.2))

    def myTool():
        print("========")
        print("holg 'b' for box selection")
        print("holg 'g' to move selected controlpoints")
        print("========")
        while True:
            # Select click
            while window.get_mouse_button(glfw.MOUSE_BUTTON_LEFT):
                mousepos = window.get_mouse_pos()
                yield
                if window.get_mouse_button(glfw.MOUSE_BUTTON_LEFT) == False:
                    P1 = mousepos[0], mousepos[1], 0
                    unprojected_points = unproject([P1], viewport=window.viewport, projection=window.projection, view=window.view)
                    model.select(unprojected_points[0])
                    print("click", window.get_mouse_pos())

                
            # Selection Rect Tool
            if window.get_key_down(glfw.KEY_B):
                mousebegin = window.get_mouse_pos()
                previous_mouse_pos = window.get_mouse_pos()
                while window.get_key_down(glfw.KEY_B):
                    mousedelta = subtract(previous_mouse_pos, window.get_mouse_pos())
                    previous_mouse_pos = window.get_mouse_pos()
                    if length(mousedelta)>0:
                        mousepos = window.get_mouse_pos()
                        P1 = mousebegin[0], mousebegin[1], 0
                        P2 = mousepos[0], mousepos[1], 0
                        P1,P2 = unproject([P1, P2], viewport=window.viewport, projection=window.projection, view=window.view)
                        rect = rect_from_corners(P1[0], P1[1], P2[0], P2[1])
                        model.perform_box_selection(rect=rect)

                        # update selection rectangle
                        selection_rect.update(rect=rect)
                    yield
                selection_rect.update(rect=(0,0,0,0))
            yield

            # Move Tool
            if window.get_key_down(glfw.KEY_G):
                mousebegin = window.get_mouse_pos()
                previous_mouse_pos = window.get_mouse_pos()
                while window.get_key_down(glfw.KEY_G):
                    mousepos = window.get_mouse_pos()
                    mousedelta = subtract(mousepos, previous_mouse_pos)
                    if length(mousedelta)>0:
                        mousepos = window.get_mouse_pos()
                        P1 = previous_mouse_pos[0], previous_mouse_pos[1], 0
                        P2 = mousepos[0], mousepos[1], 0
                        P1,P2 = unproject([P1, P2], viewport=window.viewport, projection=window.projection, view=window.view)
                        delta = subtract(P2,P1)
                        print(delta)
                        model.move(*delta)
                    previous_mouse_pos = window.get_mouse_pos()
                    yield

    tool = myTool()

    src_points = np.copy(model.controlpoints)

    @window.event
    def on_render():

        # Tools
        next(tool)

        # camera projection
        # Animate
        colors = np.full(shape=(model.controlpoints.shape[0],4), fill_value=(0.5, 0.5, 0.5, 1.0), dtype=np.float32)
        colors[model.selection] = (1,1,1,1)
        pointcloud.update(colors=colors)
        pointcloud.update(positions=model.controlpoints)
        
        img = checker(512,256)*(0.5, 0.5, 0.5, 1.0)
        h,w,c = img.shape
        src_corners = project(src_points, viewport=(0,0,w, h), projection=glm.ortho(-0.5,0.5,-0.5,0.5,-1,1), view=window.view, flip_vertical=False)
        dst_corners = project(model.controlpoints, viewport=(0,0,w, h), projection=glm.ortho(-0.5,0.5,-0.5,0.5,-1,1), view=window.view, flip_vertical=False)
        img = cornerpin(img, src_corners, dst_corners)

        imageplane.update(image=img)

        # Render
        window.clear(color=(0.1,0.1,0.1,1.0))



        imageplane.update(projection=window.projection, view=window.view)
        imageplane.render()
        pointcloud.update(projection=window.projection, view=window.view)
        pointcloud.render()
        selection_rect.update(projection=window.projection, view=window.view)
        selection_rect.render()

    window.start()

