import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
from PIL import Image
from OpenGL.error import GLError
""" how to dix resizing glitch:
https://stackoverflow.com/questions/45880238/how-to-draw-while-resizing-glfw-window
"""




image_plane = None
points = None
def init():
    global plane, points
    img = noise(128, 126)
    plane = ImagePlane(img)
    pos    = [(-0.5, -0.5, 0.0),
          ( 0.5, -0.5, 0.0),
          ( 0.5,  0.5, 0.0),
          (-0.5,  0.5, 0.0)]

    points = Points(pos)

def render():
    global plane, points

    pos    = np.array([(-0.5, -0.5, 0.0),
          ( 0.5, -0.5, 0.0),
          ( 0.5,  0.5, 0.0),
          (-0.5,  0.5, 0.0)])
    pos+=np.random.rand(*pos.shape)*0.01

    points = Points(pos)

    glClearColor(0.1, 0.1, 0.1, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    # plane.draw()
    points.draw()
    print("", flush=True, end="")


"""BOILERPLATE"""
def main():
    def on_resize(window,width,height):
        print(f"on resize {width}x{height}")
        # render()
        # glfw.swap_buffers(window)

    def on_refresh(window):
        print(f"on refresh")
        render()
        glfw.swap_buffers(window)

    # initialize glfw
    if not glfw.init():
        return

    print("creating the window")
    glfw.window_hint(glfw.DECORATED, True) # Show/Hide titlebar
    window = glfw.create_window(800, 600, "My GLFW window", None, None)

    glfw.swap_interval(1) # vsynx
    glfw.set_window_refresh_callback(window, on_refresh) # render while eventloop is blocking eg: resize
    glfw.set_framebuffer_size_callback(window, on_resize);
    
    if not window:
        glfw.terminate()
        print("cant create window")
        return

    glfw.make_context_current(window)

    init()
    print("start render loop")
    while not glfw.window_should_close(window):
        glfw.poll_events()

        """Render here"""
        render()

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()

