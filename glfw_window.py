import glfw
from OpenGL.GL import *
from typing import Tuple, List
import numpy as np
import glm

from typing import Callable
from dataclasses import dataclass, field

# from PIL import Image
# import sys
# import psutil

# @dataclass
# class Camera:
#     eye: Vec3=(0,0,0)
#     target: Vec3=(0,0,1)
#     fov: float=math.radians(90)
#     aspect: float=1.0
#     near: float=1.0
#     far: float=100.0
#     tiltshift:Vec2=(0,0)

#     def __postinit__(self):
#         """normalize parameters"""
#         self.eye = glm.vec3(self.eye)
#         self.target = glm.vec3(self.target)
#         self.tiltshift = glm.vec2(self.tiltshift)

#     @property
#     def projection(self):
#         aspect = self.aspect
#         tiltshift = glm.vec2(self.tiltshift)/self.near
#         projection = glm.frustum(-1*aspect, 1*aspect, -1+tiltshift.y, 1+tiltshift.y, self.near, self.far) # left right, bottom, top, near, far
#         return projection

#     @property
#     def view(self):
#         return glm.lookAt(self.eye, self.target, (0,1,0))

@dataclass(kw_only=True, frozen=True)
class Event:
    name:str=""
    modifiers:List = field(default_factory=list)

@dataclass(kw_only=True, frozen=True)
class RenderEvent(Event):
    name:str="render"

@dataclass(kw_only=True, frozen=True)
class KeyEvent(Event):
    key:str=""

@dataclass(kw_only=True, frozen=True)
class KeyPress(KeyEvent):
    name:str="key_press"

@dataclass(kw_only=True, frozen=True)
class KeyRelease(KeyEvent):
    name:str="key_release"

@dataclass(kw_only=True, frozen=True)
class MouseEvent(Event):
    pos:Tuple[float, float]

@dataclass(kw_only=True, frozen=True)
class MousePress(MouseEvent):
    name:str="mouse_press"
    button: int

@dataclass(kw_only=True, frozen=True)
class MouseRelease(MouseEvent):
    name:str="mouse_release"
    button: int

@dataclass(kw_only=True, frozen=True)
class MouseClick(MouseEvent):
    name:str="mouse_click"
    button: int

@dataclass(kw_only=True, frozen=True)
class MouseMove(MouseEvent):
    name:str="mouse_move"
    delta: Tuple[float,float]

@dataclass(kw_only=True, frozen=True)
class MouseDrag(MouseEvent):
    name:str="mouse_drag"
    begin: Tuple[float, float]

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

class Window:
    """
    Usage:
    class MyWindow(GLFWWindow):
        def __init__(self):
            super().__init__()
            # init your stuff here

        def draw(self):
            super().draw()
            # draw your stuff here

        def on_cursor_pos(self, window, x, y):
            print(x, y)

    window = MyWindow()
    """

    _current_window = None
    def __init__(self, width:int=720, height:int=576, title="GLFW window"):
        # Pass initial properties
        self._width, self._height = width, height
        self.__class__._current_window = self
        self._fullscreen = False

        # Init glfw window
        if not glfw.init():
            return

        glfw.window_hint(glfw.DECORATED, True) # Show/Hide titlebar
        self.window = glfw.create_window(self._width, self._height, title, None, None)
        if not self.window:
            glfw.terminate()
            print("cant create window")
            return

        glfw.make_context_current(self.window)
        glfw.swap_interval(1) # vsync
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Events
        glfw.set_window_refresh_callback(self.window, self._on_refresh) # render while eventloop is blocking eg: resize
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)

        glfw.set_cursor_pos_callback(self.window, self._on_cursor_pos)
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_char_callback(self.window, self._on_char)
        glfw.set_char_mods_callback(self.window, self._on_char_mods)
        glfw.set_cursor_enter_callback(self.window, self._on_cursor_enter)
        glfw.set_scroll_callback(self.window, self._on_scroll)

        self._mousepos = (0,0)
        self._mousebegin = (0,0)
        self._mousedelta = (0,0)
        self._is_mouse_dragging = False
        self.DRAG_THRESHOLD = 5
        self.handlers = {
            "on_input_event": [],
            "on_refresh": [],
            "on_resize": [],
            "on_render":[],
        }

        # initial resize
        self._on_resize(self.window, self.width, self.height)

    #Attach event handlers
    def on_render(self, fn:Callable):
        self.handlers["on_render"].append(fn)

    def on_input_event(self, fn:Callable):
        """register input event handler"""
        self.handlers["on_input_event"].append(fn)

    # dispatch input events
    def dispatch(self, *args, **kwargs):
        # print("event:", *args, kwargs)
        for fn in self.handlers["on_input_event"]:
            fn(*args,**kwargs)

    # GLFW INPUT CALLBACKS
    def _on_cursor_enter(self, window, entered:int):
        if entered == 1:
            self._mousebegin = self._mousepos = glfw.get_cursor_pos(window)
            self._mousedelta = (0,0)
            self.dispatch("mouseenter")
        elif entered == 0:
            self.dispatch("mouseleave")

    def _on_cursor_pos(self, window, x:float, y:float):
        self._mousedelta = subtract((x,y), self._mousepos)
        
        AnyButtonDown = any(glfw.get_mouse_button(window, button)==glfw.PRESS for button in [glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_RIGHT, glfw.MOUSE_BUTTON_MIDDLE])
        if not AnyButtonDown:
            self._mousebegin = self._mousepos
        else:
            if distance(self._mousebegin, (x,y))>self.DRAG_THRESHOLD:
                self._is_mouse_dragging = True
        self._mousepos = (x,y)
        self.dispatch("mousemove")

    def _on_mouse_button(self, window, button:int, action:int, mods:int):
        if action == glfw.PRESS:
            self.dispatch("mousepress", button, action, mods)

        elif action == glfw.RELEASE:
            self.dispatch("mouserelease", button, action, mods)
            self._is_mouse_dragging = False

    def _on_scroll(self, window, xoffset:float, yoffset:float):
        self.dispatch("scroll", xoffset, yoffset)

    def _on_key(self, window, key:int, scancode:int, action:int, mods:int):
        if key == glfw.KEY_F11  and action == glfw.PRESS:
            self.toggle_fullscreen()

        self.dispatch("key", key, scancode, action, mods)

    def _on_char(self, window, codepoint:int):
        self.dispatch("char", codepoint)

    def _on_char_mods(self, window, codepoint:int, mods:int):
        self.dispatch("char mods", codepoint, mods)

    # Get input states
    def get_mouse_pos(self)->Tuple[float, float]:
        return glfw.get_cursor_pos(self.window)

    def get_mouse_delta(self)->Tuple[float, float]:
        return self._mousedelta

    def get_mouse_begin(self)->Tuple[float, float]:
        return self._mousebegin

    def is_mouse_down(self, button: glfw.MOUSE_BUTTON_LEFT | glfw.MOUSE_BUTTON_RIGHT | glfw.MOUSE_BUTTON_MIDDLE)->bool:
        return glfw.get_mouse_button(self.window, button) == glfw.PRESS

    def is_mouse_dragging(self)->bool:
        return self._is_mouse_dragging

    def is_key_down(self, key)->bool:
        return glfw.get_key(self.window, key) == glfw.PRESS

    # Handle window refresh and resize
    def _on_refresh(self, window):
        self.render()
        glfw.swap_buffers(window)

    def _on_resize(self, window, width, height):
        glViewport(0,0,width, height)
        self._width, self._height = width, height

    # window properties
    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def aspect(self):
        return self.width / self.height

    # viewer properties
    @property
    def viewport(self):
        return (0,0,self.width, self.height)

    @property
    def projection(self):
        return glm.ortho(-self.aspect, self.aspect,-1, 1,-100,100)

    @property
    def view(self):
        return glm.translate((0,0,1))

    @classmethod
    def get_current_window(cls):
        return cls._current_window

    # ations
    def clear(self, color=(0.1, 0.1, 0.1, 1.0)):
        glClearColor(*color)
        glClear(GL_COLOR_BUFFER_BIT)

    def toggle_fullscreen(self):
        if not self._fullscreen:
            self._windowed_rect = (*glfw.get_window_pos(self.window), *glfw.get_window_size(self.window))
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            glfw.set_window_monitor(self.window, monitor, 0, 0, mode.size.width, mode.size.height, mode.refresh_rate)
            self._fullscreen = True
        else:
            x, y, w, h = self._windowed_rect
            glfw.set_window_monitor(self.window, None, x, y, w, h, glfw.DONT_CARE)
            self._fullscreen = False

    def render(self):
        for fn in self.handlers["on_render"]:
            fn()

    # callbacks?
    def start(self):
        mousepos = self.get_mouse_pos()
        mousebegin = mousepos
        lastmouse = mousepos
        mousedelta = subtract(mousepos, lastmouse)
        state = "IDLE"
        print("start render loop")
        while not glfw.window_should_close(self.window):
            self.events = []
            glfw.poll_events()

            """Render YOUR STUFF here"""
            self.render()

            # swap buffers
            glfw.swap_buffers(self.window)
            print("", end="", flush="True")

        glfw.terminate()


if __name__ == "__main__":
    import time
    class MyWindow(Window):
        def __init__(self, width:int=720, height:int=576, title="My GLFW window"):
            super().__init__(width=width, height=height, title=title)
            # init your stuff here

        def render(self):
            super().render()
            # draw your stuff here


    window = MyWindow(title="My Window")

    @window.on_input_event
    def on_input_event(event, *args, **kwargs):
        print("            event:", event)
        print("    get_mouse_pos:", window.get_mouse_pos())
        print("  get_mouse_delta:", window.get_mouse_delta())
        print("    is mouse down:", [window.is_mouse_down(button) for button in [glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_RIGHT, glfw.MOUSE_BUTTON_MIDDLE]])
        print("  get_mouse_begin:", window.get_mouse_begin())
        print("is_mouse_dragging:", window.is_mouse_dragging())
        print()

        if event == "mousepress":
            print("press")

        elif event == "mousemove":
            if window.is_mouse_dragging():
                print("drag")
            else:
                print("move")

        elif event=="mouserelease":
            if window.is_mouse_dragging():
                print("drag end")
            else:
                print("click")

        elif event=="mouseenter":
            print("mouseenter")

        elif event=="mouseleave":
            print("mouseleave")

        else:
            print("else", event, *args, kwargs)

    @window.on_render
    def render():
        pass

    window.start()