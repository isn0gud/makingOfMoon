# makingOfMoon

We copy/take inspiration from the cuda toolkit nbody example. This can be found in the local cuda installation in "samples/5_Simulations/nbody/".

Furthermore we give the same courtesy to "https://github.com/PLeLuron/nbody"

#Build
##CUDA
Cuda-9 only works with gcc/g++ 6 so CMAKE_C_COMPILER and CMAKE_CXX_COMPILER has to be set, 
as well as CUDACXX to you nvcc installation.

If gcc is used, you might need to set the correct compiler:
sudo ln -s /usr/bin/gcc-6 /usr/local/cuda/bin/gcc
sudo ln -s /usr/bin/g++-6 /usr/local/cuda/bin/g++

###glm
install glm with the package manager of the distribution you use

###glfw
install glfw with the package manager of the distribution you use

####local installation
cd lib/glfw
cmake .
make install DESTDIR=glfw_compiled

###glew

install glew with the package manager of the distribution you use.
(under ubuntu 17.10 apt installs libGLEW.so.2.0.0 to /usr/lib/x86_64-linux-gnu/.
 So you need to create a simlink manually pointing libGLEW.so -> libGLEW.so.2.0.0 )


####local installation
Download https://github.com/nigels-com/glew/releases/download/glew-2.1.0/glew-2.1.0.zip to lib/
cd lib/glew-2.1.0
make install DESTDIR=glew_compiled


#TODO

- [x] Create a inital CPU implementation with few particles
- [x] Render the particles as points with blur instead of spheres.
- [x] Create an initialization function for the planets that creates a cluster of particles that models a planet given its mass, initial linear and angular velocity and the number of particles.
- [ ] Render a video to mp4 and not to screen (running on cluster)
- [x] Port and run the simulation on CUDA
- [x] Connect the results (particle positions) of the CUDA simulation to the rendering(OpenGL) part
- [ ] Scale the CUDA implementation \cite{noauthor_gpu_nodate}

- [ ] (Hierarchical Barnes-Hut method (BH)\cite{barnes1986hierarchical})
- [ ] (Fast multipole method (FMM) \cite{greengard1987fast})
- [ ] (Particle-mesh methods \cite{darden1993particle})