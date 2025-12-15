# rccpy - A simple tool to install and run CloudComPy from R 

rccpy provides an easy one-command way to install the CloudComPy python library and execute python scripts based on CloudComPy for point cloud processing. A python module is included that wraps some functions of CloudComPy for the creation of reproducible and readable pipelines for forest science applications.

## Installation

You can install rccpy by running:

```
# install.packages("devtools")
devtools::install_github("ataeub/rccpy")
```

## Usage

To use rccpy you need the archive of the CloudComPy library, which you can get from here:

<https://www.simulation.openfields.fr/index.php/cloudcompy-downloads/3-cloudcompy-binaries/5-windows-cloudcompy-binaries/106-cloudcompy310-20240613>

Then you will need to run:

```
rccpy::install_cloudcompy("path/to/cloudcompy-archive")
```

This will take download the required miniconda installation and packages for CloudComPy and will take \~9 GB of disk space.

After this you can run python scripts with functions from CloudComPy and the cloudcompype module with:

```
rccpy::source_ccpy("path/to/cloudcompy-script.py")
```

When the package is loaded the necessary paths and imports are already set. So you do not need to import CloudComPy or cloudcompype within your python script.
