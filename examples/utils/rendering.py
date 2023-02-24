from pathlib import Path

import numpy as np
from dm_control import mjcf, viewer, mujoco
from dm_control.rl import control
from examples.other.empty import EmptyTask

env_model_path = Path(__file__).parent / "data" / "environment.xml"
quad_model = Path(__file__).parent.parent / "utils" / "quadrotor.xml"
# quad_model = Path(__file__).parent.parent / "data" / "quadrotor.xml"
task_model = Path(__file__).parent.parent / "data" / "dm" / "task.xml"


def render_xml(xml_path: Path):
    with open(xml_path) as f:
        mjcf_model = mjcf.from_file(f)
        render_model(mjcf_model)

def render_model(model: mjcf.RootElement):
    # arena = mjcf.RootElement()
    # chequered = arena.asset.add('texture', type='2d', builtin='checker', width=300,
    #                             height=300, rgb1=[.2, .3, .4], rgb2=[.3, .4, .5])
    # grid = arena.asset.add('material', name='grid', texture=chequered,
    #                        texrepeat=[5, 5], reflectance=.2)
    # arena.worldbody.add('geom', type='plane', size=[2, 2, .1], material=grid)
    # for x in [-2, 2]:
    #     arena.worldbody.add('light', pos=[x, -1, 3], dir=[-x, 1, -2])
    #
    # # Place them on a grid in the arena.
    # height = .15
    # spawn_site = arena.worldbody.add('site', pos=[0, 0, height])
    # # Attach to the arena at the spawn sites, with a free joint.
    # spawn_site.attach(model).add('freejoint')

    physics = mjcf.Physics.from_mjcf_model(model)
    task = EmptyTask()
    env = control.Environment(physics, task)
    viewer.launch(env)


if __name__ == '__main__':
    render_xml(quad_model)
