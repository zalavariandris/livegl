import glfw
from OpenGL.GL import *
from typing import Tuple
import numpy as np
import glm

from typing import Callable
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

        # initialize glfw
        if not glfw.init():
            return

        self._width, self._height = width, height

        print("creating the window")
        glfw.window_hint(glfw.DECORATED, True) # Show/Hide titlebar
        self.window = glfw.create_window(self._width, self._height, title, None, None)

        glfw.set_window_refresh_callback(self.window, self.on_refresh) # render while eventloop is blocking eg: resize
        glfw.set_framebuffer_size_callback(self.window, self.on_resize)

        glfw.set_cursor_pos_callback(self.window, self.on_cursor_pos)
        glfw.set_key_callback(self.window, self.on_key)
        # glfw.set_mouse_button_callback(self.window, self.on_mouse_button)


        if not self.window:
            glfw.terminate()
            print("cant create window")
            return
        self._fullscreen = False
        glfw.make_context_current(self.window)
        glfw.swap_interval(1) # vsync
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # handle selection
        self.mouse_pos = 0, 0
        self.mouse_down = []
        self.selection_active = False
        self.selection_start = (0,0)
        self.selection_end = (0,0)

        self.__class__._current_window = self

        self.on_resize(self.window, self.width, self.height)

        self.handlers = {
            "on_refresh": [],
            "on_resize": [],
            "on_render":[],
            "on_key_press": [],
            "on_key_release": [],
            "on_mouse_move": [],
            "on_mouse_press": [],
            "on_mouse_release": [],
            "on_mouse_drag": [],
            "on_mouse_enter": [],
            "on_mouse_leave": [],
            "on_mouse_scroll": []
        }

    def get_mouse_pos(self):
        return glfw.get_cursor_pos(self.window)

    def get_mouse_button(klass, button):
        return glfw.get_mouse_button(self.window, button)

    def on_resize(self, window, width, height):
        glViewport(0,0,width, height)
        self._width, self._height = width, height

    def dispatch(event_name:str, *args, **kwargs):
        self.handlers[event_name](*args, **kwargs)

    def event(self, fn:Callable):
        assert fn.__name__ in self.handlers, f"no {fn.__name__} event handler"
        handlers = self.handlers[fn.__name__]
        self.handlers[fn.__name__].append(fn)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def aspect(self):
        return self.width / self.height

    @property
    def viewport(self):
        return (0,0,self.width, self.height)

    @property
    def projection(self):
        return glm.ortho(-self.aspect, self.aspect,-1, 1,-100,100)

    @property
    def view(self):
        return glm.translate((0,0,1))

    def get_mouse_pos(self)->Tuple[float, float]:
        return glfw.get_cursor_pos(self.window)

    def get_mouse_button(self, button: glfw.MOUSE_BUTTON_RIGHT | glfw.MOUSE_BUTTON_LEFT | glfw.MOUSE_BUTTON_MIDDLE):
        return glfw.get_mouse_button(self.window, button) == glfw.PRESS

    def get_key_down(self, key):
        return glfw.get_key(self.window, key) == glfw.PRESS

    @classmethod
    def get_current_window(cls):
        return cls._current_window

    def dispatch(self, event:str):
        for fn in self.handlers[event]:
            fn()

    def clear(self, color=(0.1, 0.1, 0.1, 1.0)):
        glClearColor(*color)
        glClear(GL_COLOR_BUFFER_BIT)

    def render(self):
        self.dispatch("on_render")

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

    def on_key(self, window, key:int, scancode:int, action:int, mods:int):
        if key == glfw.KEY_F11  and action == glfw.PRESS:
            self.toggle_fullscreen()

    def on_refresh(self, window):
        self.render()
        glfw.swap_buffers(window)

    def on_cursor_pos(self, window, x, y):
        pass


    # def on_mouse_button(self, window, button:int, action:int, mods:int):
    #     if button == glfw.MOUSE_BUTTON_LEFT:
    #         if action == glfw.PRESS:
    #             self.selection_start = (self.mouse_x, self.mouse_y)
    #             self.selection_active = True
    #         elif action == glfw.RELEASE:
    #             self.selection_end = (self.mouse_x, self.mouse_y)
    #             self.selection_active = False
    #             self.finalize_selection()

    # def on_mouse_move(self, window, x:int, y:int):
    #     self.mouse_x = x
    #     self.mouse_y = y
    #     if self.selection_active:
    #         self.selection_end = (self.mouse_x, self.mouse_y)

    # def finalize_selection(self):
    #     if self.selection_start and self.selection_end:
    #         # Determine selected points within the selection box
    #         min_x = min(self.selection_start[0], self.selection_end[0])
    #         max_x = max(self.selection_start[0], self.selection_end[0])
    #         min_y = min(self.selection_start[1], self.selection_end[1])
    #         max_y = max(self.selection_start[1], self.selection_end[1])

    def start(self):
        print("start render loop")
        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            """Render YOUR STUFF here"""
            self.render()

            # swap buffers
            glfw.swap_buffers(self.window)
            print("", end="", flush="True")

        glfw.terminate()


if __name__ == "__main__":
    class MyWindow(GLFWWindow):
        def __init__(self, width:int=720, height:int=576, title="My GLFW window"):
            super().__init__(width=width, height=height, title=title)
            # init your stuff here

        def render(self):
            super().render()
            # draw your stuff here

        def on_cursor_pos(self, window, x, y):
            print(x, y)

    window = MyWindow(title="My Window")
    window.start()