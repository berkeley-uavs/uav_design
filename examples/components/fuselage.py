import numpy as np
from dm_control import mjcf
from examples.utils.rendering import render_model


DEFAULT_LENGTH = 0.06
DEFAULT_WIDTH = 0.035
DEFAULT_HEIGHT = 0.025
DEFAULT_MASS = 0.2


class Fuselage:

    def __init__(self,
                 size: list[int] | None = None,
                 mass: float = DEFAULT_MASS,
                 rgba: list[int] | None = None):
        self.model = mjcf.RootElement()

        self.model.default.geom.type = 'box'

        if rgba is None:
            """Random color"""
            random_state = np.random.RandomState(42)
            rgba = random_state.uniform([0, 0, 0, 1], [1, 1, 1, 1])

        if size is None:
            size = [DEFAULT_LENGTH, DEFAULT_WIDTH, DEFAULT_HEIGHT]

        self.body = self.model.worldbody.add('body')
        self.body.add('geom',
                      type="box",
                      size=size,
                      quat=[.924, 0.0, 0.0, 0.383],
                      rgba=rgba,
                      mass=mass)


if __name__ == '__main__':
    tube = Fuselage()
    render_model(tube.model)
