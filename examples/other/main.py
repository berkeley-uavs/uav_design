from pathlib import Path

from dm_control import mjcf, suite, viewer

#
# # Load a model from an MJCF XML string.
# xml_string = """
# <mujoco>
#   <worldbody>
#     <light name="top" pos="0 0 1.5"/>
#     <geom name="floor" type="plane" size="1 1 .1"/>
#     <body name="box" pos="0 0 .3">
#       <joint name="up_down" type="slide" axis="0 0 1"/>
#       <geom name="box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
#       <geom name="sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
#     </body>
#   </worldbody>
# </mujoco>
# """
# physics = mujoco.Physics.from_xml_string(xml_string)
#
# # Render the default camera view as a numpy array of pixels.
# pixels = physics.render()
#
#
#
# physics = mujoco.Physics.from_xml_string(Path("./quadrotor.xml"))
# # Render the default camera view as a numpy array of pixels.
# pixels = physics.render()
#
#
with open(Path("./quadrotor.xml")) as f:
  mjcf_model = mjcf.from_file(f)
  print(mjcf_model)





#
#
#
class Arm(object):

    def __init__(self, name):
        self.mjcf_model = mjcf.RootElement(model=name)

        self.upper_arm = self.mjcf_model.worldbody.add('body', name='upper_arm')
        self.shoulder = self.upper_arm.add('joint', name='shoulder', type='ball')
        self.upper_arm.add('geom', name='upper_arm', type='capsule',
                           pos=[0, 0, -0.15], size=[0.045, 0.15])

        self.forearm = self.upper_arm.add('body', name='forearm', pos=[0, 0, -0.3])
        self.elbow = self.forearm.add('joint', name='elbow',
                                      type='hinge', axis=[0, 1, 0])
        self.forearm.add('geom', name='forearm', type='capsule',
                         pos=[0, 0, -0.15], size=[0.045, 0.15])


class UpperBody(object):

    def __init__(self):
        self.mjcf_model = mjcf.RootElement()
        self.mjcf_model.worldbody.add(
            'geom', name='torso', type='box', size=[0.15, 0.045, 0.25])
        left_shoulder_site = self.mjcf_model.worldbody.add(
            'site', size=[1e-6] * 3, pos=[-0.15, 0, 0.25])
        right_shoulder_site = self.mjcf_model.worldbody.add(
            'site', size=[1e-6] * 3, pos=[0.15, 0, 0.25])

        self.left_arm = Arm(name='left_arm')
        left_shoulder_site.attach(self.left_arm.mjcf_model)

        self.right_arm = Arm(name='right_arm')
        right_shoulder_site.attach(self.right_arm.mjcf_model)


body = UpperBody()
physics = mjcf.Physics.from_mjcf_model(body.mjcf_model)
pixels = physics.render()
print(pixels)
#
# # Load an environment from the Control Suite.
# env = suite.load(domain_name="humanoid", task_name="stand")

# Launch the viewer application.
viewer.launch(physics)