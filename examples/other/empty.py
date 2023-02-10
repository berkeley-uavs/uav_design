from __future__ import absolute_import

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers

SUITE = containers.TaggedTasks()


@SUITE.add()
def empty(xml):
    xml_string = common.read_model(xml)
    physics = mujoco.Physics.from_xml_string(xml_string, common.ASSETS)
    task = EmptyTask()
    env = control.Environment(physics, task)
    return env


class EmptyTask(base.Task):

    def initialize_episode(self, physics):
        pass

    def get_observation(self, physics):
        return dict()

    def get_reward(self, physics):
        return 0.0

