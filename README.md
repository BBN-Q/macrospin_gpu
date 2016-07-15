# Macrospin-GPU
Using PyCUDA and/or PyOpenCL to run GPU code over many realizations of the Weiner process.

Until this is a proper package add the `src` folder to your python path.

# Dependencies

* [PyOpenCL](http://mathema.tician.de/software/pyopencl/)
* [PyOpenGL](http://pyopengl.sourceforge.net/)
* [PyCUDA](http://mathema.tician.de/software/pycuda/)
* [tdqm](https://github.com/noamraph/tqdm) (progress bar)
* PyQt4
* [jinja2](http://jinja.pocoo.org/docs/dev/) templating GPU programs

# Installation

## OS X
Installed out of the box on Windows.

## Windows
Use the Anaconda python distribution

## Linux

### NVIDIA Drivers with Prime

Tested on Linux Mint 18 with nvidia 361 drivers.  Have installed following
packages: `nvidia-361,``nvidia-opencl-icd-361`, `nvida-settings`, `nvidia-prime`
for NVIDIA support and the generic `opencl-header`, `ocl-icd-libopencl1` and
`ocl-icd-opencl-dev` for building OpenCL program and `clinfo` to see if anything
works. Need to have Nvidia card powered up using Prime.

### Build and install PyOpenCL
There is an [open issue](https://github.com/pyopencl/pyopencl/issues/81) with recent versions of PyOpenCL and gcc 4.8/4.9. Work around by checking out `v2015.1`.

    ```shell
    git clone git@github.com:pyopencl/pyopencl.git
    cd pyopencl
    git checkout v2015.1
    python configure.py --cl-inc-dir=/opt/intel/opencl-1.2-sdk-5.0.0.43/include/ --cl-lib-dir=/opt/intel/opencl-1.2-5.0.0.43/lib64/
    make install
    ```
