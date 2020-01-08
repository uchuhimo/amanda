
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

       pip install -e ".[all]"

* 
  Use poetry:

  .. code-block:: bash

       poetry install -E all

Run an example
^^^^^^^^^^^^^^

.. code-block:: bash

   make build_cc
   python src/amanda/tools/debugging/insert_debug_op.py

Usage
-----

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

   op =  amanda.Op(
       attrs={},
       input_tensors=[],
       control_dependencies=[],
       output_num=1,
   )

Update an op’s attribute:

.. code-block:: python

   op.attrs["name"] = "conv_1"

Update the input tensor of an op:

.. code-block:: python

   op.input_tensors[i] = tensor

Add a control dependency op to an op:

.. code-block:: python

   op1.add_control_dependency(op2)

Remove a control dependency op from an op:

.. code-block:: python

   op1.remove_control_dependency(op2)

Create a new graph:

.. code-block:: python

   graph = amanda.Graph(ops=[op1, op2], attrs={})

Add an op to a graph:

.. code-block:: python

   graph.add_op(op)

Remove an op from a graph:

.. code-block:: python

   graph.remove_op(op)

Development
-----------

Install git pre-commit hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pre-commit install

run tests
^^^^^^^^^

.. code-block:: bash

   python src/amanda/tests/download_model.py
   make build_cc
   pytest -n auto

Update dependencies
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   poetry update

Bump version
^^^^^^^^^^^^

.. code-block:: bash

   bumpversion minor  # major, minor, patch

Show information about installed packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   poetry show

Show dependency tree
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   dephell deps tree
   # or
   dephell deps tree pytest
