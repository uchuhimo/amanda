## Installation

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

### Run an example

```bash
make build_cc
python src/amanda/tools/debugging/insert_debug_op.py
```

## Usage

### Import a model (from TensorFlow/ONNX/...)

E.g. import from a TensorFlow checkpoint:

```python
graph = amanda.tensorflow.import_from_checkpoint(checkpoint_dir)
```

See [amanda/conversion/tensorflow.py](src/amanda/conversion/tensorflow.py) for all supported import operations in TensorFlow.

### Export a model (to TensorFlow/ONNX/...)

E.g. export to a TensorFlow checkpoint:

```python
amanda.tensorflow.export_to_checkpoint(graph, checkpoint_dir)
```

See [amanda/conversion/tensorflow.py](src/amanda/conversion/tensorflow.py) for all supported export operations in TensorFlow.

### All supported import/export modules

| Framework | Module |
| --- | --- |
| TensorFlow | [amanda.tensorflow](src/amanda/conversion/tensorflow.py) |
| ONNX | [amanda.onnx](src/amanda/conversion/onnx.py) |
| MMdnn | [amanda.mmdnn](src/amanda/conversion/mmdnn.py) |

### modify the graph

See [amanda/graph.py](src/amanda/graph.py) for all Graph/Op APIs.

Import amanda:

```python
import amanda
```

Create a new op and its output tensors:

```python
op =  amanda.Op(
    attrs={},
    input_tensors=[],
    control_dependencies=[],
    output_num=1,
)
```

Update an op’s attribute:

```python
op.attrs["name"] = "conv_1"
```

Update the input tensor of an op:

```python
op.input_tensors[i] = tensor
```

Add a control dependency op to an op:

```python
op1.add_control_dependency(op2)
```

Remove a control dependency op from an op:

```python
op1.remove_control_dependency(op2)
```

Create a new graph:

```python
graph = amanda.Graph(ops=[op1, op2], attrs={})
```

Add an op to a graph:

```python
graph.add_op(op)
```

Remove an op from a graph:

```python
graph.remove_op(op)
```

## Development

### Install git pre-commit hooks

```bash
pre-commit install
```

### run tests

```bash
python src/amanda/tests/download_model.py
make build_cc
pytest -n auto
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

### Update dependencies

```bash
poetry update
```

### Bump version

```bash
bumpversion minor  # major, minor, patch
```
