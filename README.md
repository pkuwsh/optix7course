# Siggraph 2019 Optix 7 Course Tutorial Code

adsf

# Building the Code

This code was intentionally written with minimal dependencies,
requiring pretty much only CMake (as a build system), your favorite
compiler (tested with Visual Studio under Windows, and GCC under
Linux), and of course the Optix 7 SDK (including CUDA 10.1 and the
most recent NVIDIA developer driver).

## Dependencies

- a compiler
	- On windows, tested with Visual Studio 2017 community edition
	- On Linux, tested with Ubuntu 18 and Ubuntu 19 default gcc installs
- CUDA 10.1
	- Download from developer.nvidia.com
	- on Linux, suggest to put `/usr/local/cuda/bin` into your `PATH`
- latest NVIDIA developer driver that comes with the SDK
	- download link to go online during siggraph 2019
- Optix 7 SDK
	- download link to go online during siggraph 2019
	- on linux, suggest to put the optix lib directory into your `LD_LIBRARY_PATH`
	- on windows, suggest to add the optix lib directory to the system environment `PATH` variable

The literally only *external* library we use is GLFW for windowing, and
even this one we actually build on the fly under Windows, so installing
it is only required under Linux. 

Detailed steps below:

## Building under Linux

- Install required packages

	sudo apt install libglfw3-dev cmake-curses-gui

- Clone the code

	git clone http://gitlab.com/ingowald/optix7course
	cd optix7course

- create (and enter) a build directory

	mkdir build
	cd build

- configure with cmake

	ccmake ..

- and build

	make

## Buiding under Windows

- Install Required Packages
	- see above: CUDA 10.1, Optix 7 SDK, latest driver, and cmake
- download or clone the source repository
- Open `CMake GUI` from your start menu
	- point "source directory" to the downloaded source directory
	- point "build directory" do <source directory>/build (agree to create this directory when prompted)
	- click 'configure'. If CUDA, SDK, and compiler are all properly installed this should enable the 'generate' button. If not, make sure all dependencies are properly installed, "clear cache", and re-configure.
	- click 'generate' (this creates a visual studio project and solutions)
	- click 'open project' (this should open the project in visual studio)


# Examples Overview Overview
	
## Example 1: Hello World 

This is how this should look like in Linux:
!(./example01_helloOptix/ex01-linux.png)

And here, in Windows:
!(./example01_helloOptix/ex01-windows.png)

## Example 1: Hello World 

!(./example02_pipelineAndRayGen/ex02-output.png)
## Example 1: Hello World 

!(./example03_inGLFWindow/ex03-linux.png)
## Example 1: Hello World 

!(./example03_inGLFWindow/ex03-windows.png)
## Example 1: Hello World 

!(./example04_firstTriangleMesh/ex04.png)
## Example 1: Hello World 

!(./example05_firstSBTData/ex05.png)
## Example 1: Hello World 

!(./example06_multipleObjects/ex06.png)
## Example 1: Hello World 

!(./example07_firstRealModel/ex07.png)
## Example 1: Hello World 

!(./example08_addingTextures/ex08.png)
## Example 1: Hello World 

!(./example09_shadowRays/ex09.png)
## Example 1: Hello World 

!(./example10_softShadows/ex10.png)
## Example 1: Hello World 

