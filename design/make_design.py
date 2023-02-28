from examples.utils.rendering import render_model, render_xml
from examples.components.fuselage import Fuselage
from examples.components.tubes import Arm
from examples.components.thruster import Thruster
from dm_control import mjcf
import json
from pathlib import Path

env_model_path = Path(__file__).parent / "environment.xml"
quad_model = Path(__file__).parent /  "example_quad.json"

class Design:
    body: str
    model: mjcf.RootElement

    def __init__(self):
        self.model = mjcf.from_file(env_model_path)
        self.body = self.model.worldbody.add('body')

    def parse_grid(self, path):
        """TODO: figure out how to determine quats from grid representation and add sensors/sites/actuators"""
        # load json with grid data
        f = open(path)
        data = json.load(f)
        f.close()

        quats = [
            [.924, 0.0, 0.0, 0.483],
            [.483, 0.0, 0.0, 0.924],
            [-.483, 0.0, 0.0, 0.924],
            [.924, 0.0, 0.0, -0.483],
        ]

        nodes = data['NODES']
        ind = 0
        for component in nodes.keys():
            if component == 'core':
                Fuselage(body=self.body, pos=nodes[component], quat=[1.0, 0.0, 0.0, 0])
            elif 'arm' in component:
                Arm(body=self.body, pos=nodes[component], quat=quats[ind])
                ind += 1
            elif 'thruster' in component:
                Thruster(body=self.body, pos=nodes[component])


if __name__ == '__main__':
    design = Design()
    design.parse_grid(quad_model)
    render_model(design.model)
