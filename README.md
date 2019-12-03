## Create a virtual environment

```bash
conda env create -f environment.yml
source activate mmx
```

## Install dependencies

There are two options:

- Use pip:
    ```bash
    pip install -e ".[all]"
    ```
- Use poetry:
    ```bash
    poetry install -E all
    ```

## Update dependencies

```bash
poetry update
```

## Install git pre-commit hooks

```bash
pre-commit install
```
