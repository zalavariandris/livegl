import glfw # type: ignore
from OpenGL.GL import * # type: ignore
import OpenGL.GL.shaders # type: ignore
from typing import Callable, Any

from dataclasses import dataclass
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
class PointSelector:
    def __init__(self, window:int, candidates:np.ndarray):
        self.window = window
        self.candidates = candidates

        # store mouse state
        self.mousepos = -1,-1
        self.dragstart = None  # Variable to store the starting position of the selection box
        self.dragend = None    # Variable to store the ending position of the selection box
        self.mousedown = None
        self.selected_indices = []

    def get_mouse_pos(self):
        return glfw.get_cursor_pos(self.window)

    def get_mouse_button(self, button:glfw.MOUSE_BUTTON_LEFT | glfw.MOUSE_BUTTON_RIGHT | glfw.MOUSE_BUTTON_MIDDLE):
        return glfw.get_mouse_button(self.window, button)

    def update(self):
        # Check if left mouse button is pressed
        

        IsMouseMoving = self.mousepos != self.get_mouse_pos()
        MouseReleased = self.mousedown is True and self.get_mouse_button(MOUSE_BUTTON_LEFT) is False
        self.mousedown = self.get_mouse_button(MOUSE_BUTTON_LEFT)
        self.mousepos = self.get_mouse_pos()
        self.mousedown = IsMouseDown

        IsDragging = IsMouseMoving and IsMouseDown

        if IsDragging:
            pass

        if MouseReleased:
            print("mouse released")

        # def rect_from_corners(x1, y1, x2, y2)->Tuple:
        #     x = min(x1, x2)
        #     y = min(y1, y2)
        #     w = max(x1, x2)-x
        #     h = max(y1, y2)-y
        #     return x,y,w,h

        # if self.drag_start is None and self.get_mouse_button(glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
        #     self.mousepos = self.get_mouse_pos()
        #     print("mouse down", self.mousepos)
        #     # Get mouse cursor position
        #     # Update selection start position
        #     self.drag_start = self.mousepos

        # elif self.drag_start is not None and self.get_mouse_button(glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
        #     if self.mousepos != self.get_mouse_pos():
        #         self.mousepos = self.get_mouse_pos()
        #         print("mouse drag", self.mousepos)
        #         self.drag_end = self.mousepos
        #         # rect = rect_from_corners(*self.drag_start, *self.mousepos)
        #         # self.perform_box_selection(self.candidates, rect)
        # elif self.drag_start is not None and self.drag_end is None and self.get_mouse_button(glfw.MOUSE_BUTTON_LEFT) == glfw.RELEASE:
        #     print("click")
        #     self.drag_start = None
        # elif self.drag_start is not None and self.get_mouse_button(glfw.MOUSE_BUTTON_LEFT) == glfw.RELEASE:
        #     self.mousepos = self.get_mouse_pos()
        #     self.drag_start = None
        #     print("mouse drag release", self.mousepos)

        # self.mousepos = self.get_mouse_pos()
        return self.selected_indices

    def perform_box_selection(self, points, rect:Tuple[float, float, float, float]):
        x, y, w, h = rect
        # Convert cursor position from window coordinates to OpenGL coordinates
        width, height = glfw.get_framebuffer_size(self.window)
        x = (2 * x / width) - 1
        y = 1 - (2 * y / height)
        w = w/width*2
        h = h/height*2

        for i, point in enumerate(self.candidates):
            if x <= point[0] <= x+width and y <= point[1] <= y+width:
                self.selected_indices.append(i)


class Mouse:
    def __init__(self):

        self.current_buttons = []
        self.prev_buttons = self.current_buttons
        self.prev_pos = (-1, -1)
        self.current_pos = self.prev_pos
        self.dragging = False

    def update(self, current_pos:Tuple[float, float], current_buttons:List[bool]):
        # print(current_buttons)
        self.current_buttons = current_buttons
        self.current_pos = current_pos

        # Check for button click events
        for i in range(len(self.current_buttons)):
            if self.current_buttons[i] and not self.prev_buttons[i]:
                self.mouse_clicked(i)
                self.dragging = True

        # Check for mouse position change
        if self.current_pos != self.prev_pos:
            self.mouse_hover(self.current_pos[0], self.current_pos[1])

            # Check for drag event
            if self.dragging:
                self.mouse_drag(self.current_pos[0], self.current_pos[1])

        # Check if mouse is released
        if not any(self.current_buttons):
            if self.dragging:
                self.dragging = False

        self.prev_buttons = self.current_buttons
        self.prev_pos = self.current_pos

    def mouse_clicked(self, button):
        # Callback for mouse click event
        print(f"Mouse clicked: Button {button}")

    def mouse_hover(self, x, y):
        # Callback for mouse hover event
        print(f"Mouse hover: x={x}, y={y}")

    def mouse_drag(self, x, y):
        # Callback for mouse drag event
        print(f"Mouse drag: x={x}, y={y}")

class MyWindow(GLFWWindow):
    def __init__(self):
        super().__init__()
        coords = np.array([(-0.5, -0.5, 0.0),
              ( 0.5, -0.5, 0.0),
              ( 0.5,  0.5, 0.0),
              (-0.5,  0.5, 0.0)], dtype=np.float32)
        coords+=(np.random.rand(*coords.shape)-np.array([0.5, 0.5, 0.5]))*0.1
        self.points = Mesh.points(coords=coords)

        self.mouse = Mouse()
        
        
    def render(self):
        super().render()
        self.imageplane = Mesh.imagePlane(image=noise(128, 126))
        # self.imageplane.render()


        self.points.render()
        self.mouse.update(glfw.get_cursor_pos(self.window), [True if glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) else False])
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

