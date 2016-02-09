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

### Intel CPU OpenCL libraries
Tested on Linux Mint 17.2

Mostly following the instructions from [this Intel forum
post](https://software.intel.com/en-us/forums/opencl/topic/390630#comment-1832283)

1. Download the [Intel OpenCL Code Builder for
Ubnutu](https://software.intel.com/en-us/articles/opencl-drivers#ubuntu64) which
bizarrely comes with a whole bunch of rpms and the install scripts will demand
Ubuntu 12.04 and fail. Extract and move into the folder

    ```shell
    tar xzf intel_code_builder_for_opencl_2015_ubuntu_5.0.0.43_x64.tgz
    cd intel_code_builder_for_opencl_2015_ubuntu_5.0.0.43_x64
    ```

1. Install some rpm to deb tools

    ```shell
    sudo apt-get install -y rpm alien libnuma1sudo apt-get install -y rpm alien libnuma1
    ```

1. Install the Intel public key

    ```shell
    sudo rpm --import PUBLIC_KEY.PUB
    ```

1. Repackage the rpms to debs and install

    ```shell
    fakeroot alien --to-deb opencl-1.2-base-5.0.0.43-1.x86_64.rpm
    fakeroot alien --to-deb opencl-1.2-intel-cpu-5.0.0.43-1.x86_64.rpm
    fakeroot alien --to-deb opencl-1.2-devel-5.0.0.43-1.x86_64.rpm
    fakeroot alien --to-deb opencl-1.2-intel-devel-5.0.0.43-1.x86_64.rpm
    sudo dpkg -i opencl-1.2-base_5.0.0.43-2_amd64.deb
    sudo dpkg -i opencl-1.2-intel-cpu_5.0.0.43-2_amd64.deb
    sudo dpkg -i opencl-1.2-devel_5.0.0.43-2_amd64.deb
    sudo dpkg -i opencl-1.2-intel-devel_5.0.0.43-2_amd64.deb
    ```

1. Add the library to the search path by adding a file to `/etc/ld.so.donc.d` called something like `intelOpenCL.conf` and adding the install directory from the above packages.  Should be something like `/opt/intel/opencl-1.2.5.0.0.43/lib64`. Then rehash with `sudo ldconfig`.
1. Add the Intel icd loader by linking

    ```shell
    sudo ln /opt/intel/opencl-1.2-5.0.0.43/etc/intel64.icd /etc/OpenCL/vendors/intel64.icd
    ```
### Nvidia Drivers with Prime
Tested on Linux Mint 17.3 with nvidia 352.63 drivers.  Have installed
`nvidia-352,``nvidia-opencl-icd-352`, `nvida-settings`, `nvidia-prime`,
`opencl-headers`. Need to have Nvidia card powered up using Prime. Then need to
initialize nvida_uvm kernel module by running something as root. E.g.

```shell
~/Downloads $ curl https://codeload.github.com/hpc12/tools/tar.gz/master | tar xvfz -
~/Downloads $ cd tools-master
/tools-master $ make
/tools-master $ sudo ./print_devices
```

Now you'll be able to run OpenCL programs without root.  This might be related
to `nvidia-352-modprobe` not being available for Ubuntu 14.04. I tried manually
installing `nvidia-modprobe` from
ftp://download.nvidia.com/XFree86/nvidia-modprobe/ and it didn't help.


### Build and install PyOpenCL
There is an [open issue](https://github.com/pyopencl/pyopencl/issues/81) with recent versions of PyOpenCL and gcc 4.8/4.9. Work around by checking out `v2015.1`.

    ```shell
    git clone git@github.com:pyopencl/pyopencl.git
    cd pyopencl
    git checkout v2015.1
    python configure.py --cl-inc-dir=/opt/intel/opencl-1.2-sdk-5.0.0.43/include/ --cl-lib-dir=/opt/intel/opencl-1.2-5.0.0.43/lib64/
    make install
    ```
