
Development
-----------

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

Install git pre-commit hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pre-commit install

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

run tests
^^^^^^^^^

.. code-block:: bash

   pytest -n auto

Run an example
^^^^^^^^^^^^^^

.. code-block:: bash

   python src/amanda/tests/test_tf_import_export.py
