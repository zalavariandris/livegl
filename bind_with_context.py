from contextlib import contextmanager
from OpenGL.GL import * # type: ignore
from typing import Iterator

@contextmanager
def bind_vertex_array(vao_id:int)->Iterator[int]:
    try:
        glBindVertexArray(vao_id)
        yield vao_id
    finally:
        glBindVertexArray(0)
        if gl_error := glGetError():
            print(f"OpenGL error in bind_vertex_array: {gl_error}")

@contextmanager
def bind_buffer(target: GL_ARRAY_BUFFER | GL_ELEMENT_ARRAY_BUFFER, buffer_id:int)->Iterator[int]:
    try:
        glBindBuffer(target, buffer_id)
        yield buffer_id
    finally:
        glBindBuffer(target, 0)
        if gl_error := glGetError():
            print(f"OpenGL error in bind_buffer: {gl_error}")

@contextmanager
def bind_texture(target: GL_TEXTURE_2D, texture_id:int)->Iterator[int]:
    try:
        glBindTexture(target, texture_id)
        yield texture_id
    finally:
        glBindTexture(target, 0)
        if gl_error := glGetError():
            print(f"OpenGL error in bind_texture: {gl_error}")

@contextmanager
def use_program(program_id:int)->Iterator[int]:
    try:
        glUseProgram(program_id)
        yield program_id
    finally:
        glUseProgram(0)
        if gl_error := glGetError():
            print(f"OpenGL error in use_program: {gl_error}")

@contextmanager
def bind_framebuffer_object(fbo_id: int) -> Iterator[int]:
    """
    Context manager for binding an existing Framebuffer Object (FBO) in OpenGL.

    Parameters:
        fbo_id (int): The ID of the FBO to bind.

    Yields:
        int: The ID of the bound Framebuffer Object.
    """
    try:
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_id)
        yield fbo_id
    finally:
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if gl_error := glGetError():
            print(f"OpenGL error in bind_framebuffer_object context manager: {gl_error}")
