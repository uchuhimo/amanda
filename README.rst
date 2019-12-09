
Development
-----------

Create a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   conda env create -f environment.yml
   source activate mmx

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

Install git pre-commit hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pre-commit install
