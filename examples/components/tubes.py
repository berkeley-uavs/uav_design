import numpy as np
from dm_control import mjcf
from examples.utils.rendering import render_model

DEFAULT_LENGTH = 0.05
DEFAULT_WIDTH = 0.01
DEFAULT_HEIGHT = 0.0025


class Arm:

    def __init__(self,
                 body,
                 pos: list[float],
                 quat: list[float],
                 size: list[int] | None = None,
                 rgba: list[int] | None = None):

        if rgba is None:
            """Random color"""
            random_state = np.random.RandomState(42)
            rgba = random_state.uniform([0, 0, 0, 1], [1, 1, 1, 1])

        if size is None:
            size = [DEFAULT_LENGTH, DEFAULT_WIDTH, DEFAULT_HEIGHT]

        body.add('geom',
                 type="box",
                 pos=pos,
                 size=size,
                 quat=quat,
                 rgba=rgba,
                 mass=.025)


if __name__ == '__main__':
    tube = Arm()
    render_model(tube.model)
