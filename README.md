# uav_design

# Setup

## Prerequisites
[pdm](https://github.com/pdm-project/pdm) (Make sure you run pdm version larger than 2.1.4)

Python 3.11

### Install

To install dependencies run:

```bash
pdm install
```



## Usage
Check and run the python scripts that you find in the  `examples` folder. Use the code as examples form where you can copy snippets and create your own environment.

### Rendering from monolithic xml
Check out and run `examples/utils/rendering.py`

### Rendering from python data structures
Check out and run the python files in `examples/components/`


# Working with IDEs

With PEP 582, dependencies will be installed into `__pypackages__` directory under the project root. With PEP 582 enabled globally, you can also use the project interpreter to run scripts directly.
Check [pdm documentation](https://pdm.fming.dev/latest/usage/pep582/) on PEP 582.


**PYCHARM**
Add `__pypackages__/3.11/lib` and `src` folders to your PYTHONPATH. With PyCharm you can simple right click on the folders and select `Mark Directory as` - `Source folder`.

**VSCODE**

To configure VSCode to support PEP 582, open `.vscode/settings.json` (create one if it does not exist) and add the following entries:
```json
{
  "python.autoComplete.extraPaths": ["__pypackages__/3.11/lib"],
  "python.analysis.extraPaths": ["__pypackages__/3.11/lib"]
}
```
