import math
from typing import Tuple
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

def ToolTemplate( window):
    print("INITALIYE MY TOOL")
    state = MouseState.IDLE
    mousepos = glfw.get_cursor_pos(window)
    mousebegin = mousepos
    selection = []
    
    while True:
        mousedelta = subtract(mousepos, glfw.get_cursor_pos(window))
        mousepos = glfw.get_cursor_pos(window)
        leftbutton = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS


        if state == MouseState.IDLE:
            if leftbutton:
                state = MouseState.DOWN
                mousebegin = glfw.get_cursor_pos(window)
                yield MouseEvent.PRESSED

            if length(mousedelta)>0:
                yield MouseEvent.MOVED, mousedelta

        if state==MouseState.DOWN: 
            if leftbutton == False:
                state = MouseState.IDLE

            if distance(mousepos, mousebegin) > 5:
                 state = MouseState.DRAG
                 yield MouseEvent.DRAG_STARTED, mousepos
                 mousebegin = mousepos

            elif leftbutton is False:
                yield MouseEvent.CLICKED, mousepos
                state = MouseState.IDLE

        if state==MouseState.DRAG:
            if length(mousedelta)>0:
                yield MouseEvent.DRAGGED, subtract(mousebegin, mousepos)
            if leftbutton is False:
                yield MouseEvent.DRAG_FINISHED
                state = MouseState.IDLE
        yield None

import glfw
def SelectionTool(positions, window):
    def mousepos():
        return glfw.get_cursor_pos(window)

    def leftbutton():
        return glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    # mousebegin = mousepos
    selection = []

    def perform_box_selection(positions, rect):
        selection = []
        for i, P in enumerate(positions):
            if rect_contains(rect, P):
                selection.append(i)
        return selection
    
    while True:
        if leftbutton():
            mousebegin = glfw.get_cursor_pos(window)
            while leftbutton():
                rect = rect_from_corners(*mousebegin, *mousepos())
                selection = perform_box_selection(positions, rect)
                yield selection
        yield None

def MoveTool(positions, window):
    def mousepos():
        return glfw.get_cursor_pos(window)

    def leftbutton():
        return glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS

    prev_pos = mousepos()
    while True:
        mousedelta = subtract(prev_pos, mousepos())
        prev_pos = mousepos()
        yield mousedelta #subtract(mousebegin, mousepos())
        if leftbutton():
            break