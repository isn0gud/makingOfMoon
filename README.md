# makingOfMoon

We copy/take inspiration from the cuda toolkit nbody example. This can be found in the local cuda installation in "samples/5_Simulations/nbody/".

Furthermore we give the same courtesy to "https://github.com/PLeLuron/nbody"


# TODO

- [x] Create a inital CPU implementation with few particles
- [ ] Render the particles as points with blur instead of spheres.
	- [ ] Combine inital CPU impl with the particle rendering
- [ ] Create an initialization function for the planets that creates a cluster of particles that models a planet given its mass, initial linear and angular velocity and the number of particles.
- [ ] Render a video to mp4 and not to screen (running on cluster)
- [ ] Port and run the simulation on CUDA
- [ ] Connect the results (particle positions) of the CUDA simulation to the rendering(OpenGL) part
- [ ] Scale the CUDA implementation \cite{noauthor_gpu_nodate}

- [ ] (Hierarchical Barnes-Hut method (BH)\cite{barnes1986hierarchical})
- [ ] (Fast multipole method (FMM) \cite{greengard1987fast})
- [ ] (Particle-mesh methods \cite{darden1993particle})
