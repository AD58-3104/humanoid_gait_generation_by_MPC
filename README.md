# humanoid_gait_generation_by_MPC
Here are some examples of walking pattern generators using Model predictive control.

List of paper which is implemented in this repository.
- P. -b. Wieber, "Trajectory Free Linear Model Predictive Control for Stable Walking in the Presence of Strong Perturbations," 2006 6th IEEE-RAS International Conference on Humanoid Robots, Genova, Italy, 2006, pp. 137-142, doi: 10.1109/ICHR.2006.321375.


## Required library
-  gnuplot-cpp https://github.com/martinruenz/gnuplot-cpp.git
- Eigen3.4 https://eigen.tuxfamily.org/index.php?title=3.4
- osqp https://osqp.org/
- osqp-eigen https://github.com/robotology/osqp-eigen.git

## How to run 
1. install Eigen3.4 & osqp & osqp-eigen
2. cd [your target directory] 
3. mkdir build
4. cmake .. 
5. make
6. ./exmple
