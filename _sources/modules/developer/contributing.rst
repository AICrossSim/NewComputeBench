Contributing & Maintaining Docs
================================

This guide covers how to build, preview, and extend the documentation.


Setting Up the Doc Environment
--------------------------------

In addition to the main :doc:`../getting_started/installation`, install the Sphinx toolchain.
Dependencies are declared in ``pyproject.toml`` and pinned in ``uv.lock``.

.. code-block:: bash

   # with uv (reproducible, uses uv.lock — recommended)
   uv sync

   # or with pip
   pip install sphinx sphinx-book-theme sphinx-autobuild


Building the Documentation
---------------------------

.. code-block:: bash

   cd docs
   make html

The output is written to ``docs/build/html/``. Open ``docs/build/html/index.html``
in a browser to preview the result.

For a live-reloading preview during editing, install ``sphinx-autobuild``:

.. code-block:: bash

   pip install sphinx-autobuild
   sphinx-autobuild docs/source docs/build/html


Adding a New Page
------------------

1. Create a new ``.rst`` file in the appropriate directory under ``docs/source/modules/``.

2. Register it in the parent ``index.rst`` by adding it to the ``.. toctree::`` directive.

   For example, to add a tutorial ``docs/source/modules/tutorials/simulations/my_tutorial.rst``,
   edit ``docs/source/modules/tutorials/index.rst``:

   .. code-block:: rst

      .. toctree::
         :maxdepth: 1
         :caption: Model Behaviour-Level Simulation

         simulations/my_tutorial

3. Write the page content in RST. Follow the conventions below.


RST Conventions
----------------

Heading hierarchy
~~~~~~~~~~~~~~~~~

Use the following underline characters consistently across all files:

.. code-block:: rst

   Page Title
   ==========

   Major Section
   -------------

   Subsection
   ~~~~~~~~~~

   Sub-subsection
   ^^^^^^^^^^^^^^

Code blocks
~~~~~~~~~~~

Always specify the language for syntax highlighting:

.. code-block:: rst

   .. code-block:: bash

      conda activate new-compute
      python run.py hf-gen --model_name AICrossSim/clm-60m

Admonitions
~~~~~~~~~~~

Use standard Sphinx admonitions:

.. code-block:: rst

   .. note::

      Extra information the reader should be aware of.

   .. warning::

      Something the reader must be careful about.

   .. tip::

      A useful tip or successful result to highlight.

   .. admonition:: Custom Title

      Use this for admonitions that need a descriptive title.

Cross-references
~~~~~~~~~~~~~~~~

- Link to another page: ``:doc:`relative/path/to/file``` (no ``.rst`` extension)
- Link to a labelled section: ``:ref:`label-name```
- External link: ``` `Link text <https://example.com>`_ ```

Tables
~~~~~~

Use ``.. list-table::`` for tables with complex cell content:

.. code-block:: rst

   .. list-table::
      :header-rows: 1
      :widths: 40 60

      * - Column A
        - Column B
      * - Row 1
        - Content


Documentation Structure
------------------------

.. code-block:: text

   docs/
   ├── Makefile
   └── source/
       ├── conf.py                          # Sphinx configuration
       ├── index.rst                        # Top-level page (intro, roadmap, what's new)
       ├── _static/
       │   └── images/                      # All images used in the docs
       └── modules/
           ├── getting_started/
           │   ├── installation.rst
           │   ├── quickstart.rst
           │   └── models.rst
           ├── tutorials/
           │   ├── pretraining/
           │   │   └── llm_pretrain_eval.rst
           │   └── simulations/
           │       ├── bitflip_clm.rst
           │       ├── bitflip_lora.rst
           │       ├── onn_roberta.rst
           │       ├── onn_clm.rst
           │       ├── snn_roberta.rst
           │       ├── pim_roberta.rst
           │       ├── pim_vit.rst
           │       └── mase_triton.rst
           ├── developer/
           │   ├── contributing.rst         # This file
           │   └── sim_guidelines.rst
           └── changelog.rst
