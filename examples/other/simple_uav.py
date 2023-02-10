from pathlib import Path

from dm_control import mjcf, suite, viewer, mujoco
from dm_control.rl import control
from empty import EmptyTask

with open(Path("./quadrotor.xml")) as f:
    mjcf_model = mjcf.from_file(f)

    physics = mjcf.Physics.from_mjcf_model(mjcf_model)

    task = EmptyTask()
    env = control.Environment(physics, task)
    viewer.launch(env)
