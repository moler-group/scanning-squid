# Scanning SQUID measurement and control
### Moler Group, Stanford University

scanning-squid is an instrument control and data acquisition package for scanning SQUID (Superconducting QUantum Interference Device) microscopy. It is based on the [QCoDeS](http://qcodes.github.io/Qcodes/) framework, and is designed to run in either a standalone [Jupyter Notebook](http://jupyter.org/index.html), or in [Jupyter Lab](http://jupyterlab.readthedocs.io/en/stable/).

### Project homepage: [scanning-squid.readthedocs.io](https://scanning-squid.readthedocs.io/en/latest/).
### Contact: Logan Bishop-Van Horn (lbvh [at] stanford [dot] edu).

#### Installation
scanning-squid is not yet packaged for [pip](https://pypi.org/) or [conda](https://conda.io/docs/). It is recommended that you set up a `conda env` in which to run scanning-squid by following the steps below. This will install all of the packages on which scanning-squid depends.

##### Windows
- Download and install [Anaconda](https://www.anaconda.com/download/#windows) (the latest Python 3 version).
- Download [environment.yml](https://github.com/moler-group/scanning-squid/blob/master/environment.yml) from the scanning-squid repository
- Launch an Anaconda Prompt (start typing anaconda in the start menu and click on Anaconda Prompt)
- In the Anaconda Prompt, navigate to the directory containing environment.yml (`cd <path-to-directory-containing-environment-file>`)
- Run the following two commands in the Anaconda Prompt:
  - `conda env create -f environment.yml`
  - `activate scanning-squid`
  
##### Mac
- Download and install [Anaconda](https://www.anaconda.com/download/#macos) (the latest Python 3 version).
- Download [environment.yml](https://github.com/moler-group/scanning-squid/blob/master/environment.yml) from the scanning-squid repository
- Launch a Terminal.
- In the Terminal, navigate to the directory containing environment.yml (`cd <path-to-directory-containing-environment-file>`)
- Run the following two commands in the Terminal:
  - `conda env create -f environment.yml`
  - `source activate scanning-squid`
