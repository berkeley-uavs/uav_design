import numpy as np
from dm_control import mjcf
from examples.utils.rendering import render_model

BODY_RADIUS = 0.1
BODY_SIZE = (BODY_RADIUS, BODY_RADIUS, BODY_RADIUS / 2)


class Hub:

    def __init__(self,
                 n_sides: int = 4,
                 top: bool = False,
                 bottom: bool = False,
                 rgba: list[int] | None = None):

        self.model = mjcf.RootElement()

        if rgba is None:
            """Random color"""
            random_state = np.random.RandomState(42)
            rgba = random_state.uniform([0, 0, 0, 1], [1, 1, 1, 1])

        self.model.compiler.angle = 'radian'

        # Make the hub geom.
        self.model.worldbody.add(
            'geom', name='hub', type='box', size=BODY_SIZE, rgba=rgba)

        for i in range(n_sides):
            theta = 2 * i * np.pi / n_sides
            side_pos = BODY_RADIUS * np.array([np.cos(theta), np.sin(theta), 0])
            side_site = self.model.worldbody.add('site', pos=side_pos, euler=[0, 0, theta])
            # leg = Leg(length=BODY_RADIUS, rgba=rgba)
            # side_site.attach(leg.model)


if __name__ == '__main__':
    hub = Hub()
    render_model(hub.model)
