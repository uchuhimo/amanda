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
    pip install -e ".[dev]"
    ```

- Use poetry:

    ```bash
    poetry install
    ```

- Build in debug mode for C++:

    ```bash
    DEBUG=1 poetry install
    ```

### Run an example

```bash
make build_cc
amanda-download tf --model vgg16 --root-dir downloads
python src/amanda/tools/debugging/insert_debug_op_tensorflow.py
```

## Usage

### CLI

The usage of `amanda`:

```
Usage: amanda [OPTIONS] [TOOL_ARGS]...

Options:
  -i, --import [amanda_proto|amanda_yaml|tensorflow_pbtxt|tensorflow_checkpoint|tensorflow_saved_model|torchscript|onnx_model|onnx_graph|mmdnn]
                                  Type of the imported model.  [required]
  -f, --from PATH                 Path of the imported model.  [required]
  -e, --export [amanda_proto|amanda_yaml|tensorflow_pbtxt|tensorflow_checkpoint|tensorflow_saved_model|torchscript|onnx_model|onnx_graph|mmdnn]
                                  Type of the exported model.  [required]
  -t, --to PATH                   Path of the exported model.  [required]
  -ns, --namespace TEXT           Namespace of the graph instrumented by the
                                  tool.

  -T, --tool TEXT                 Fully qualified name of the tool.
  --help                          Show this message and exit.
```

E.g. use a tool to insert debugging ops into a TensorFlow graph from a checkpoint:

```bash
# download the checkpoint
amanda-download tf --model vgg16 --root-dir downloads
# run the debugging tool
amanda --import tensorflow_checkpoint --from downloads/model/vgg16/imagenet_vgg16.ckpt \
       --export tensorflow_checkpoint --to tmp/modified_model/vgg16/imagenet_vgg16.ckpt \
       --namespace amanda/tensorflow \
       --tool amanda.tools.debugging.insert_debug_op_tensorflow.DebuggingTool
# run the modified model
amanda-run amanda.tools.debugging.insert_debug_op_tensorflow.run_model --model-dir tmp/modified_model/vgg16
```

The modified model will be saved into `tmp/modified_model/vgg16`, and the debugging information will be stored into `tmp/debug_info/vgg16`.

E.g. convert a TensorFlow model to an Amanda graph:

```bash
amanda --import tensorflow_checkpoint --from downloads/model/vgg16/imagenet_vgg16.ckpt \
       --export amanda_yaml --to tmp/amanda_graph/vgg16/imagenet_vgg16
```

The Amanda graph will be saved into `tmp/amanda_graph/vgg16`.

E.g. convert an Amanda graph to a TensorFlow model:

```bash
amanda --import amanda_yaml --from tmp/amanda_graph/vgg16/imagenet_vgg16 \
       --export tensorflow_checkpoint --to tmp/tf_model/vgg16/imagenet_vgg16.ckpt
```

The TensorFlow model will be saved into `tmp/tf_model/vgg16`.

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
| PyTorch | [amanda.pytorch](src/amanda/conversion/pytorch.py) |
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
op =  amanda.create_op(
    type="Conv2D",
    attrs={},
    inputs=["input", "filter"],
    outputs=["output"],
)
```

Update an op’s attribute:

```python
op.attrs["data_format"] = "NHWC"
```

Create a new graph:

```python
graph = amanda.create_graph(
    ops=[op1, op2],
    edges=[edge],
    attrs={},
)
```

Add an op to a graph:

```python
graph.add_op(op)
```

Remove an op from a graph:

```python
graph.remove_op(op)
```

Add an edge to a graph:

```bash
graph.create_edge(op1.output_port("output"), op2.input_port("input"))
```

Remove an edge from a graph:

```bash
graph.remove_edge(edge)
```

## Development

### CMake

```bash
cmake -DCMAKE_PREFIX_PATH=`python -c "import torch;print(torch.utils.cmake_prefix_path)"`\;`python -m pybind11 --cmakedir` -S cc -B build
cd build
make
```

For VSCode:

```bash
cmake -DCMAKE_PREFIX_PATH=`python -c "import torch;print(torch.utils.cmake_prefix_path)"`\;`python -m pybind11 --cmakedir` -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE -DCMAKE_BUILD_TYPE=Release -S cc -B .vscode/build -G Ninja
```

### Install git pre-commit hooks

```bash
pre-commit install
```

### Update git pre-commit hooks

```bash
pre-commit autoupdate
```

### run tests

```bash
amanda-download all --root-dir downloads
make build_cc
KMP_AFFINITY=disabled pytest -n 2
```

Run quick tests only:

```bash
KMP_AFFINITY=disabled pytest -n 2 -m "not slow"
```

Run a single test:

```bash
pytest src/amanda/tests/test_op.py -k "test_new_op"
```

### Show information about installed packages

```bash
poetry show --latest
# or
poetry show --outdated
```

### Show dependency tree

```bash
poetry show --tree
# or
poetry show --tree pytest
```

### Update dependencies

```bash
poetry update
```

### Bump version

```bash
bumpversion minor  # major, minor, patch
```

### Measure code coverage

```bash
coverage run -m pytest
coverage html
```
