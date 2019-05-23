# FFTX
This is the source tree for FFTX.

## Building FFTX
### Prerequisites

#### C Compiler and Build Tools

FFTX builds on Linux/Unix with **gcc** and **make**, on Windows it builds with **Visual Studio**.

#### FFTW

You can download FFTW from [fftw.org](http://fftw.org/download.html).

On Windows install FFTW in **C:\FFTW3** if possible.  Be sure to follow the instructions in **README-WINDOWS** in the FFTW
root directory about using the **lib** command to create the .lib files.

#### CMake

FFTX needs a version of CMake no older than 3.8, but try to use the most recent version, which is currently 3.14.

You can download CMake from [cmake.org](https://cmake.org/download/), where there are source trees and Windows
installers.  Be wary of pre-built Linux packages from other sources, as they are often out of date.

On super computers the default CMake version may be quite old, requiring you to explicitly load the module for the latest version.

### Building on Linux and Other Unix-Like Systems

From the top level FFTX directory:
```
mkdir build
cd build
cmake ..
make
make test
```

#### Using a Custom FFTW Installation

There are two command line variables to CMake for FFTX that specify the include dir and library for a custom FFTW installation,
**FFTW_INCLUDE_DIR** and **FFTW_LIBRARY**.  Use them in the CMake command line as in the following example where FFTW is
installed in ``~/fftw3``:

```
cmake -D FFTW_INCLUDE_DIR=~/fftw3/include -D FFTW_LIBRARY=~/fftw3/lib/libfftw3.a ..
```

#### Release and Debug Builds

Use the **CMAKE_BUILD_TYPE** command line variable to explicitly control the FFTX build type.  The value can be either
**Debug** or **Release**, as in:

```
cmake -D CMAKE_BUILD_TYPE=Debug ..
```


### Building on Windows

In the top level FFTX directory, make a directory called **build**.  From a terminal window in the **build**
directory enter one of the following commands, depending on your version of Visual Studio.  See the 
[CMake documentation](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html#visual-studio-generators)
if your version isn't shown here.

```
cmake -G "Visual Studio 14 2015" -A x64 ..

cmake -G "Visual Studio 15 2017" -A x64 ..

cmake -G "Visual Studio 16 2019" -A x64 ..
```

When CMake is finished, open the new **FFTX.sln** with Visual Studio to build the projects in the code tree.


### Building on Summit

The following minimal script will set up your environment to build with the [instructions for Linux](#building-on-linux).

```
module load gcc/8.1.1
module load fftw/3.3.8
module load cmake/3.13.4

export CC='/sw/summit/gcc/8.1.1/bin/gcc'
```

CMake looks for the **CC** environment variable to override the default C compiler, which is **/usr/bin/cc** on Summit,
even if you explicitly load a gcc module.

