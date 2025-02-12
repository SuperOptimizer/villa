
# Installation

1. Activate your virtual environment. Instructions for `direnv`.

```sh
echo 'layout pyenv 3.12.5' > .envrc 
direnv allow
```

2. Install dependencies. `scrollcase` is installed as an editable package.

```sh
pip install -r requirements.txt
```

3. Install the `OCP CAD Viewer` VSCode estension to view BRep models displayed with `show()`. 

# Project structure

- `src/`: scrollcase source code
- `scripts/`: select example scripts

Within `src/`:
- `case.py`: Case construction with `build123d`
- `mesh.py`: Mesh processing
- `alignment.py`: Smallest enclosing cylinder
- `smallest_circle.py`: Helper utilities for Welzl's algorithm



## Avoid pushing meshes in Jupyter notebooks
```
git config filter.strip-notebook-output.clean 'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR'
```

# Examples

See `scripts/` directory.