from typing import Tuple, List
from dataclasses import dataclass, field
import numpy as np
from enum import Enum
import math

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
    MOVED = 6

@dataclass(kw_only=True)
class MouseTool:
    mousepos: Tuple[float, float] = None
    leftbutton: bool = False
    buttonbegin: int=None
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

            if length(self.mousedelta)>0:
                self.event(MouseEvent.MOVED)

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
        elif event == MouseEvent.MOVED:
            pass

@dataclass
class SelectionTool(MouseTool):
    candidates: np.ndarray
    _selection: List[int]=field(default_factory=list)

    def event(self, event):
        if event == MouseEvent.MOVED:
            print()

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

    @property
    def selection(self):
        return self._selection


class MoveTool(MouseTool):
    def event(self, event):
        if event == MouseEvent.MOVED:
            print(self.mousedelta)