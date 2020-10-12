
Installation
------------

Create a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   conda env create -f environment.yml
   source activate amanda

Install dependencies
^^^^^^^^^^^^^^^^^^^^

There are two options:


* 
  Use pip:

  .. code-block:: bash

       pip install -e ".[dev]"

* 
  Use poetry:

  .. code-block:: bash

       poetry install

Run an example
^^^^^^^^^^^^^^

.. code-block:: bash

   make build_cc
   amanda-download tf --model vgg16 --root-dir downloads
   python src/amanda/tools/debugging/insert_debug_op_tensorflow.py

Usage
-----

CLI
^^^

The usage of ``amanda``\ :

.. code-block::

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

E.g. use a tool to insert debugging ops into a TensorFlow graph from a checkpoint:

.. code-block:: bash

   # download the checkpoint
   amanda-download tf --model vgg16 --root-dir downloads
   # run the debugging tool
   amanda --import tensorflow_checkpoint --from downloads/model/vgg16/imagenet_vgg16.ckpt \
          --export tensorflow_checkpoint --to tmp/modified_model/vgg16/imagenet_vgg16.ckpt \
          --namespace amanda/tensorflow \
          --tool amanda.tools.debugging.insert_debug_op_tensorflow.DebuggingTool
   # run the modified model
   amanda-run amanda.tools.debugging.insert_debug_op_tensorflow.run_model --model-dir tmp/modified_model/vgg16

The modified model will be saved into ``tmp/modified_model/vgg16``\ , and the debugging information will be stored into ``tmp/debug_info/vgg16``.

E.g. convert a TensorFlow model to an Amanda graph:

.. code-block:: bash

   amanda --import tensorflow_checkpoint --from downloads/model/vgg16/imagenet_vgg16.ckpt \
          --export amanda_yaml --to tmp/amanda_graph/vgg16/imagenet_vgg16

The Amanda graph will be saved into ``tmp/amanda_graph/vgg16``.

E.g. convert an Amanda graph to a TensorFlow model:

.. code-block:: bash

   amanda --import amanda_yaml --from tmp/amanda_graph/vgg16/imagenet_vgg16 \
          --export tensorflow_checkpoint --to tmp/tf_model/vgg16/imagenet_vgg16.ckpt

The TensorFlow model will be saved into ``tmp/tf_model/vgg16``.

Import a model (from TensorFlow/ONNX/...)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

E.g. import from a TensorFlow checkpoint:

.. code-block:: python

   graph = amanda.tensorflow.import_from_checkpoint(checkpoint_dir)

See `amanda/conversion/tensorflow.py <src/amanda/conversion/tensorflow.py>`_ for all supported import operations in TensorFlow.

Export a model (to TensorFlow/ONNX/...)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

E.g. export to a TensorFlow checkpoint:

.. code-block:: python

   amanda.tensorflow.export_to_checkpoint(graph, checkpoint_dir)

See `amanda/conversion/tensorflow.py <src/amanda/conversion/tensorflow.py>`_ for all supported export operations in TensorFlow.

All supported import/export modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Framework
     - Module
   * - TensorFlow
     - `amanda.tensorflow <src/amanda/conversion/tensorflow.py>`_
   * - PyTorch
     - `amanda.pytorch <src/amanda/conversion/pytorch.py>`_
   * - ONNX
     - `amanda.onnx <src/amanda/conversion/onnx.py>`_
   * - MMdnn
     - `amanda.mmdnn <src/amanda/conversion/mmdnn.py>`_


modify the graph
^^^^^^^^^^^^^^^^

See `amanda/graph.py <src/amanda/graph.py>`_ for all Graph/Op APIs.

Import amanda:

.. code-block:: python

   import amanda

Create a new op and its output tensors:

.. code-block:: python

   op =  amanda.create_op(
       type="Conv2D",
       attrs={},
       inputs=["input", "filter"],
       outputs=["output"],
   )

Update an op’s attribute:

.. code-block:: python

   op.attrs["data_format"] = "NHWC"

Create a new graph:

.. code-block:: python

   graph = amanda.create_graph(
       ops=[op1, op2],
       edges=[edge],
       attrs={},
   )

Add an op to a graph:

.. code-block:: python

   graph.add_op(op)

Remove an op from a graph:

.. code-block:: python

   graph.remove_op(op)

Add an edge to a graph:

.. code-block:: bash

   graph.create_edge(op1.output_port("output"), op2.input_port("input"))

Remove an edge from a graph:

.. code-block:: bash

   graph.remove_edge(edge)

Development
-----------

Install git pre-commit hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pre-commit install

run tests
^^^^^^^^^

.. code-block:: bash

   amanda-download all --root-dir downloads
   make build_cc
   KMP_AFFINITY=disabled pytest -n 2

Run quick tests only:

.. code-block:: bash

   KMP_AFFINITY=disabled pytest -n 2 -m "not slow"

Run a single test:

.. code-block:: bash

   pytest src/amanda/tests/test_op.py -k "test_new_op"

Show information about installed packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   poetry show --latest
   # or
   poetry show --outdated

Show dependency tree
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   poetry show --tree
   # or
   poetry show --tree pytest

Update dependencies
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   poetry update

Bump version
^^^^^^^^^^^^

.. code-block:: bash

   bumpversion minor  # major, minor, patch

Measure code coverage
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   coverage run -m pytest
   coverage html
