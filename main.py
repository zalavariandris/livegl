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

def noise(w:int, h:int)->np.ndarray:
    return np.random.rand(w,h,4).astype(np.float32)

from mesh import Mesh
import time



from enum import Enum
import math

def subtract(V1, V2):
    return V1[0]-V2[0], V1[1]-V1[1]

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

class MouseState(Enum):
    IDLE = 1
    DRAG = 2
    DOWN = 3

class MouseEvent(Enum):
    PRESSED = 1
    CLICKED = 2
    DRAG_STARTED = 3
    DRAGGED = 4
    DRAG_FINISHED = 5

@dataclass(kw_only=True)
class MouseTool:
    mousepos: Tuple[float, float] = None
    leftbutton: bool = False
    mousebegin: Tuple[float, float] = None
    mousedelta: Tuple[float, float] = (0, 0)
    _state: MouseState = MouseState.IDLE

    @property
    def state(self):
        return self._state
    
    @state.setter
    def state(self, new_state):
        self._state = new_state
        # print(new_state)

    def update(self, mousepos: Tuple[float, float], leftbutton:bool):
        self.mousedelta = subtract(mousepos, self.mousepos) if self.mousepos is not None else (0, 0)
        self.mousepos = mousepos
        self.leftbutton = leftbutton

        if self.state == MouseState.IDLE:
            if self.leftbutton:
                self.mousebegin = self.mousepos
                self.state = MouseState.DOWN
                self.event(MouseEvent.PRESSED)

        if self.state==MouseState.DOWN: 
            if self.leftbutton == False:
                self.state = MouseState.IDLE

            if distance(self.mousepos, self.mousebegin) > 5:
                 self.state = MouseState.DRAG
                 self.event(MouseEvent.DRAG_STARTED)
                 self.mousebegin = self.mousepos

            elif self.leftbutton is False:
                self.event(MouseEvent.CLICKED)
                self.state = MouseState.IDLE

        if self.state==MouseState.DRAG:
            if length(self.mousedelta)>0:
                self.event(MouseEvent.DRAGGED)
            if self.leftbutton is False:
                self.event(MouseEvent.DRAG_FINISHED)
                self.state = MouseState.IDLE

    def event(self, event:MouseEvent):
        print(event)
        if event == MouseEvent.PRESSED:
            pass
        elif event == MouseEvent.CLICKED:
            pass
        elif event == MouseEvent.DRAG_STARTED:
            pass

        elif event == MouseEvent.DRAGGED:
            pass
        elif event == MouseEvent.DRAG_FINISHED:
            pass

@dataclass
class SelectionTool(MouseTool):
    candidates: np.ndarray
    _selection: List[int]=field(default_factory=list)

    def event(self, event):
        if event == MouseEvent.PRESSED:
            pass

        elif event == MouseEvent.CLICKED:
            for i, P in enumerate(self.candidates):
                if distance(P, self.mousepos) < 5:
                    self._selection = [i]
                    print(self._selection)

        elif event == MouseEvent.DRAG_STARTED:
            pass

        elif event == MouseEvent.DRAGGED:
            self._selection = []
            for i, P in enumerate(self.candidates):
                rect = rect_from_corners(*self.mousebegin, *self.mousepos)
                if rect_contains(rect, P):
                    self._selection.append(i)
            print(self._selection)

        elif event == MouseEvent.DRAG_FINISHED:
            print("select points in ", self.mousebegin, self.mousepos)

    def selection(self):
        return []

def to_screen_space(points:np.ndarray, viewport:Tuple[float, float, float, float]):
    x, y, w, h = viewport
    return (points[:,:2]+1.0)/2*(w, h)-(x, y)

class MyWindow(GLFWWindow):
    def __init__(self):
        super().__init__()
        coords = np.array([(-0.5, -0.5, 0.0),
              ( 0.5, -0.5, 0.0),
              ( 0.5,  0.5, 0.0),
              (-0.5,  0.5, 0.0)], dtype=np.float32)
        coords+=(np.random.rand(*coords.shape)-np.array([0.5, 0.5, 0.5]))*0.1
        colors = [(1,0,0,1),(0,1,0,1),(0.3,.3,1,1),(1,1,0,1)]
        self.points = Mesh.points(positions=coords, colors=colors)

        screen_coords = to_screen_space(coords, (0, 0, self.width, self.height))
        self.selectiontool = SelectionTool(candidates=screen_coords)
        
        
    def render(self):
        super().render()
        self.imageplane = Mesh.imagePlane(image=noise(128, 126))
        
        self.selectiontool.update(mousepos=glfw.get_cursor_pos(self.window), leftbutton=True if glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) else False)

        self.selectiontool.selection()
        self.points.color = np.array( [(1,1,1,1),(1,1,1,1),(1,1,1,1),(1,1,1,1)], dtype=np.float32 )
        self.points.render()
        # selection = self.selector.update()



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

