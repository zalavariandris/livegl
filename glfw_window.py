import glfw
from OpenGL.GL import *
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
    def __init__(self, width:int=720, height:int=576):

        # initialize glfw
        if not glfw.init():
            return

        print("creating the window")
        glfw.window_hint(glfw.DECORATED, True) # Show/Hide titlebar
        self.window = glfw.create_window(width, height, "My GLFW window", None, None)

        glfw.set_window_refresh_callback(self.window, self.on_refresh) # render while eventloop is blocking eg: resize
        glfw.set_framebuffer_size_callback(self.window, self.on_resize);
        glfw.set_cursor_pos_callback(self.window, self.on_cursor_pos)
        glfw.set_key_callback(self.window, self.on_key);
        # glfw.set_mouse_button_callback()

        if not self.window:
            glfw.terminate()
            print("cant create window")
            return
        self._fullscreen = False
        glfw.make_context_current(self.window)
        glfw.swap_interval(1) # vsynx
        glEnable(GL_PROGRAM_POINT_SIZE)

    def draw(self):
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
        self.draw()
        glfw.swap_buffers(window)

    def on_cursor_pos(self, window, x, y):
        pass

    def start(self):
        print("start render loop")
        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            """Render YOUR STUFF here"""
            self.draw()

            # swap buffers
            glfw.swap_buffers(self.window)

        try:
            glfw.terminate()
        except GLFWError as err:
            print("EEEEEEEEEEEEEEE")