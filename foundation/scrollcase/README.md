
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

# Usage

## Examples

See `scripts/` directory.

Simple case generation example at `scripts/example_basic_case.ipynb`.

## Case construction

Case as viewed in the OCP CAD Viewer:

<img width="600" alt="image" src="https://github.com/user-attachments/assets/cd32c255-e841-4cdb-8a43-0bdbd0284553" />

Enabling transparency to check clearances etc.:

<img width="600" alt="image" src="https://github.com/user-attachments/assets/495dcffd-c353-496e-b086-b02d5a0bc729" />

## Customizing the case

The case is customized using the ScrollCase class:

<img width="600" alt="image" src="https://github.com/user-attachments/assets/d478fc5f-d35e-4726-ac6d-09b44c65b227" />


### Distribution of case sizes

Can be generated with `scratch/02-05_scroll-sizes.ipynb`.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/c957bf06-5891-4335-912b-e2bb35e2d76d" />



# Other notes
## Avoid pushing meshes in Jupyter notebooks
```
git config filter.strip-notebook-output.clean 'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR'
```

