## Development

### Create a virtual environment

```bash
conda env create -f environment.yml
source activate amanda
```

### Install dependencies

There are two options:

- Use pip:

    ```bash
    pip install -e ".[all]"
    ```

- Use poetry:

    ```bash
    poetry install -E all
    ```

### Install git pre-commit hooks

```bash
pre-commit install
```

### Update dependencies

```bash
poetry update
```

### Bump version

```bash
bumpversion minor  # major, minor, patch
```

### Show information about installed packages

```bash
poetry show
```

### Show dependency tree

```bash
dephell deps tree
# or
dephell deps tree pytest
```

### run tests

```bash
python src/amanda/tests/download_model.py
pytest -n auto
```

### Run an example

```bash
python src/amanda/tests/test_tf_import_export.py
```
