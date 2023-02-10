from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app

from dm_control import viewer
from empty import  empty

def main(argv):
    if len(argv) < 2:
        raise Exception("Required positional argument missing. Example Usage: python xml-explore.py cheetah.xml")

    xml_file = argv[1]
    task_kwargs = dict(
        xml=xml_file
    )

    def loader():
        env = empty.SUITE["empty"](**task_kwargs)
        return env

    viewer.launch(loader)


if __name__ == '__main__':
    app.run(main)