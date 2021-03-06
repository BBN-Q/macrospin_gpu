# Macrospin-GPU

![macrospin_gpu](doc/COST-M-Vis.png)

Use PyOpenCL to run GPU code over many realizations of the Weiner process. A
convenient Jinja2-based templating system is used to allow easy specification of
physical geometry, magnetic properties, arbitrarily many spin-torque terms, and
more.

# Dependencies

* [PyOpenCL](http://mathema.tician.de/software/pyopencl/) for computation
* [PyOpenGL](http://pyopengl.sourceforge.net/) for visualization
* [tdqm](https://github.com/noamraph/tqdm) for progress bar
* [jinja2](http://jinja.pocoo.org/docs/dev/) templating GPU kernels

# Installation

Installing `macrospin_gpu` itself is quite simple. It's best to ensure that
PyOpenCL is properly installed (as described below) before continuing. At that
point you should simply be able to run:

```shell
git clone git@github.com:BBN-Q/macrospin_gpu.git
cd macrospin_gpu
pip install -e .
```

Try out some of the examples in `scripts/` to get started. Eventually I'll add
some legitimate documentation, but until that point the examples should be sufficient.
OpenGL visualizations can be seen in the `scripts/costm-visualization.py` script.

## OS X
Installing PyOpenCL with pip can be unreliable. It is best to clone the github
repository and work from there. Instead of working from the lastest and greatest
version you may want to check out a recently tagged release:

```shell
git clone git@github.com:pyopencl/pyopencl.git
cd pyopencl
git checkout v2016.2.1
python configure.py
make -j4
make install
```
In more recent builds of PyOpenCL, OpenGL interoperation should come for free.
Older versions required configuring with the `--cl-enable-gl` option.

It may be possible to use PyOpenCL from Anaconda, but I have not had a chance to
test this path.

## Windows
Use the Anaconda python distribution.

## Linux

PyOpenCL is not available from Anaconda so there is a bit more configuration to
be done on Linux.

### NVIDIA Drivers with Prime

Tested on Linux Mint 18.1 with nvidia 378 drivers.  Have installed following
packages: `nvidia-378`, `nvidia-opencl-icd-378`, `nvida-settings`,
`nvidia-prime` and `nvidia-modprobe-361`  for NVIDIA support and the generic
`opencl-headers`, `ocl-icd-libopencl1` and `ocl-icd-opencl-dev` for building
OpenCL program and `clinfo` to see if anything works. Need to have Nvidia card
powered up using Prime.

### Build and install PyOpenCL

Again, manually building is recommended. On modern Linux distributions when
using Anaconda Python there may be a conflict between the system `libstdc++` and
the one Anaconda ship. If you see errors with "undefined symbols" when trying to
import `pyopencl`, you may have to using Anaconda gcc to build PyOpenCL or
rename the Anaconda `libstdc++`. See this [SO
answer](http://stackoverflow.com/a/43244137/5445278) for details.

```shell
git clone git@github.com:pyopencl/pyopencl.git
cd pyopencl
git checkout v2016.2.1
./configure.py
make -j4
make install
```

## Funding

This software is based in part upon work supported by the Office of the Director
of National Intelligence (ODNI), Intelligence Advanced Research Projects
Activity (IARPA), via contract W911NF-14-C0089. The views and conclusions
contained herein are those of the authors and should not be interpreted as
necessarily representing the official policies or endorsements, either expressed
or implied, of the ODNI, IARPA, or the U.S. Government.
