import glfw
from OpenGL.GL import *
from typing import Tuple
# import numpy as np
# from PIL import Image
# import sys
# import psutil

class GLFWWindow:
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
        glfw.set_mouse_button_callback(self.window, self.on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self.on_mouse_move)

        if not self.window:
            glfw.terminate()
            print("cant create window")
            return
        self._fullscreen = False
        glfw.make_context_current(self.window)
        glfw.swap_interval(1) # vsync
        glEnable(GL_PROGRAM_POINT_SIZE)

        # handle selection
        self.mouse_pos = 0, 0
        self.mouse_down = []
        self.selection_active = False
        self.selection_start = (0,0)
        self.selection_end = (0,0)

        self.__class__._current_window = self

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def get_mouse_pos(self)->Tuple[float, float]:
        return glfw.get_cursor_pos(self.window)

    def get_mouse_button(self, button: glfw.MOUSE_BUTTON_RIGHT | glfw.MOUSE_BUTTON_LEFT | glfw.MOUSE_BUTTON_MIDDLE):
        return glfw.get_mouse_button(self.window, button) == glfw.PRESS

    @classmethod
    def get_current_window(cls):
        return cls._current_window

    def render(self):
        glClearColor(0.1, 0.1, 0.1, 1.0)
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

    def on_key(self, window, key:int, scancode:int, action:int, mods:int):
        if key == glfw.KEY_F11  and action == glfw.PRESS:
            self.toggle_fullscreen()

    def on_resize(self, window, width, height):
        glViewport( 0,0, width,height)

    def on_refresh(self, window):
        self.render()
        glfw.swap_buffers(window)

    def on_mouse_button(self, window, button:int, action:int, mods:int):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.selection_start = (self.mouse_x, self.mouse_y)
                self.selection_active = True
            elif action == glfw.RELEASE:
                self.selection_end = (self.mouse_x, self.mouse_y)
                self.selection_active = False
                self.finalize_selection()

    def on_mouse_move(self, window, x:int, y:int):
        self.mouse_x = x
        self.mouse_y = y
        if self.selection_active:
            self.selection_end = (self.mouse_x, self.mouse_y)

    def finalize_selection(self):
        if self.selection_start and self.selection_end:
            # Determine selected points within the selection box
            min_x = min(self.selection_start[0], self.selection_end[0])
            max_x = max(self.selection_start[0], self.selection_end[0])
            min_y = min(self.selection_start[1], self.selection_end[1])
            max_y = max(self.selection_start[1], self.selection_end[1])

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