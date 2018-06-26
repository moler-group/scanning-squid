.. _getting-started:

Getting Started
===============

`scanning-squid <https://github.com/moler-group/scanning-squid>`_ is an instrument control and data acquisition package for scanning SQUID (Superconducting QUantum Interference Device) microscopy. It is based on the `QCoDeS <http://qcodes.github.io/Qcodes/>`_ framework, and is designed to run in either a standalone `Jupyter Notebook <http://jupyter.org/index.html>`_, or in `Jupyter Lab <http://jupyterlab.readthedocs.io/en/stable/>`_.

**Contact: Logan Bishop-Van Horn (lbvh [at] stanford [dot] edu).**

Installation
------------

scanning-squid is not yet packaged for `pip <https://pypi.org/>`_ or `conda <https://conda.io/docs/>`_. It is recommended that you set up a :code:`conda env` in which to run scanning-squid by following the steps below. This will install all of the packages on which scanning-squid depends.

Windows
~~~~~~~

	- Download and install `Anaconda <https://www.anaconda.com/download/#windows>`_ (the latest Python 3 version).
	- Download `environment.yml <https://github.com/moler-group/scanning-squid/blob/master/environment.yml>`_ from the scanning-squid repository
	- Launch an Anaconda Prompt (start typing anaconda in the start menu and click on Anaconda Prompt)
	- In the Anaconda Prompt, navigate to the directory containing environment.yml (:code:`cd <path-to-directory-containing-environment-file>`)
	- Run the following two commands in the Anaconda Prompt:

  		- :code:`conda env create -f environment.yml`
  		- :code:`activate scanning-squid`

Mac
~~~

	- Download and install `Anaconda <https://www.anaconda.com/download/#macos>`_ (the latest Python 3 version).
	- Download `environment.yml <https://github.com/moler-group/scanning-squid/blob/master/environment.yml>`_ from the scanning-squid repository
	- Launch a Terminal.
	- In the Terminal, navigate to the directory containing environment.yml (:code:`cd <path-to-directory-containing-environment-file>`)
	- Run the following two commands in the Terminal:

  		- :code:`conda env create -f environment.yml`
  		- :code:`source activate scanning-squid`

After cloning the `scanning-squid repository <https://github.com/moler-group/scanning-squid>`_, to run scanning-squid from a Windows (Mac) machine, open the Anaconda Prompt (Terminal) and run :code:`activate scanning-squid` (:code:`source activate scanning-squid`), and launch a :code:`jupyter notebook` or :code:`jupyter lab`.

Documentation
~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 4

   /pages/microscope
   /pages/config-files
   /examples/PhysicalUnits
   /pages/measurements
   /pages/plots
   /pages/utils
   /examples/DataSetExample
