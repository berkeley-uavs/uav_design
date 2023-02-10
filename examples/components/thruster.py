import numpy as np
from dm_control import mjcf
from examples.utils.rendering import render_model

DEFAULT_DIAMETER = 1


class Thruster:

    def __init__(self,
                 length: int = DEFAULT_DIAMETER,
                 rgba: list[int] | None = None):
        self.model = mjcf.RootElement()

        # Defaults:
        self.model.default.joint.damping = 2
        self.model.default.joint.type = 'hinge'
        self.model.default.geom.type = 'capsule'

        if rgba is None:
            """Random color"""
            random_state = np.random.RandomState(42)
            rgba = random_state.uniform([0, 0, 0, 1], [1, 1, 1, 1])

        self.model.default.geom.rgba = rgba  # Continued below...

        # Thigh:
        self.thigh = self.model.worldbody.add('body')
        self.hip = self.thigh.add('joint', axis=[0, 0, 1])
        self.thigh.add('geom', fromto=[0, 0, 0, length, 0, 0], size=[length / 4])

        # Hip:
        # self.shin = self.thigh.add('body', pos=[length, 0, 0])
        # self.knee = self.shin.add('joint', axis=[0, 1, 0])
        # self.shin.add('geom', fromto=[0, 0, 0, 0, 0, -length], size=[length / 5])

        # Position actuators:
        # self.model.actuator.add('position', joint=self.hip, kp=10)
        # self.model.actuator.add('position', joint=self.knee, kp=10)


if __name__ == '__main__':
    tube = Thruster()
    render_model(tube.model)
