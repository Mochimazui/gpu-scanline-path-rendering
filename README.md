# Efficient GPU Path Rendering Using Scanline Rasterization

SIGGRAPH Asia 2016 project.

## Pre-built Binary 

./x64/release/gpu-scanline.exe. Tested on 64 bit Windows 10. Copy to ./working_directory to run. <br/>
Built with Visual Studio 2013 and CUDA 7.5. <br/>
Requires a NVIDIA graphics card with CUDA sm_50 and OpenGL 4.5 support.

Right button: move. <br/>
Mouse wheel: scale. <br/>
Left button: draw.

## Build Dependency

* Visual Studio 2013/2015
* CUDA 7.5/8.0, lower versions may work as well.
* Thrust

----
Open source code and pre-built binaries included in /3rd

* [SDL 2.0.3](https://www.libsdl.org/) for basic window system and UI.
* [Boost 1.60.0](http://www.boost.org/) for command line options.

Libraries are built on a 64-bit Windows 10 system with Visual Studio 2013.
You may need to download or build these libraries on your own system.

----
Other included open source code

* [Modern GPU](https://nvlabs.github.io/moderngpu/) for segmented sort.
* [glm 0.9.6.3](http://www.g-truc.net/) for vector and matrix.
* [stb](https://github.com/nothings/stb) for image and font.
* [rapidxml](https://github.com/dwd/rapidxml) for SVG parsing.

----
Code generator used

* [glLoadGen](https://bitbucket.org/alfonse/glloadgen/wiki/Home) for OpenGL functions.

## Build

Open in Visual Studio. <br/>
Check if "Properties -> CUDA C/C++ -> Device -> Code Generation" matches your device. <br/>
Then build.

## Run

* Start in Visual Studio: set "Debugging -> Working Directory" to $(SolutionDir)working_directory. <br/>
* Start in explorer or command line: copy exe file to working_directory, or create shortcut. <br/>

The program loads ./vg_default.cfg by default. Run with --help or check cmd files in working directory for more detail.

## Data

RVG files in ./input/rvg from [MPVG](http://w3.impa.br/~diego/projects/GanEtAl14/). <br/>
Works on SVG files with a subset of features (see the paper for details).

## Driver Issue

At the time we release the code, we are using driver 368.81, and everything runs well.

We found the behavior of gl_SampleMask in GLSL has changed since NVIDIA driver version 368.22.
While the old behaviour was inconsistent with OpenGL standard, we assume it was a driver bug,
or a result of incorrect graphics card configuration.
Using drivers earlier than this version may get incorrect rendering results.
