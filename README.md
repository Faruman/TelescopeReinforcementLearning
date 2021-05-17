# Off-line AO-RL repositories

### 1) Usage

1) Data to learn off-line RL should be in data folder.
2) Jupyter notebook has some exploratory analysis.
3) Train with train_offline.py. Change list rep_paths = ["/gpfs/scratch/bsc28/bsc28921/outputs/replay_9_2m_10x10_replay"] with data path.

#### 2) Package requirements

For exploratory data analysis:

+ numpy
+ matplotlib
+ jupyter notebook

For training with train_offline.py
+ torch (tested with 1.2.0 in PowerPC architecture)
+ numpy (tested with 1.16.6 in PowerPC architecture)
+ tensorboardX
+ python (tested with version 3.7.6)

#### 3) Brief explanation of the problem.

Closed-loop Adaptive Optics:
a) Light wavefront from a distant star arrives as a planar wave into the atmosphere.
b) The atmosphere changes the shape of the wavefront due to refraction and the fact that the atmosphere is not homogeneous.
c) A closed-loop AO systems corrects this perturbation. It is formed by a deformable mirror that corrects the wavefront shape, a wavefront sensor that interprets the residual perturbation in the wavefront, a real time controller that issues the actions the DM has to take and a camera that constructs the image of the target.

#### The wavefront sensor

The wavefront sensor is formed by a lenslet array. Each lens focus the light on a subaperture. If the light is focused on the center of the subaperture the wavefront is planar. Otherwise some correction has to be made.

In a closed-loop, the wavefront sensor looks at the wavefront after the correction by the DM, hence it is looking at the residual error left to correct.

![image1](sample_images/wfs.png)

Image Source: https://www.researchgate.net/figure/Principle-of-the-Shack-Hartmann-wavefront-sensor-top-undistorted-wavefront-and_fig1_258516995

Usually, to understand the pixel information of each subaperture, the center of gravity (CoG) method is used. After producing the CoG for each subaperture a measurement vector, m, is produced that has the x and y coordinates of each subaperture.

m = (x_1, x_2, ... y_1, y_2 ...)

#### The real time controller

A typical controller calculates the command at each iteration with c = R · m. Where R is the command matrix, which is the pseudoinverse of the interaction matrix, D, calculated with the least squares approach minimising |m - D · c| < ε.

Due to error source such as delay, aliasing or noise the prediction from the command matrx is not perfect. Hence, the command is computed as an integrator with gain:

C_t = C_t-1 + g R m_t 

Where t indicates timestep.

#### The RL controller

To learn a RL controller we are using a policy, π, learned by Soft Actor Critic method to compute residual actions in the command at each timestep:

C_t = C_t-1 + g R m_t + a

a ~ π(s)

Where s is a state formed by past commands and current wavefront sensor measurements. The past commands are necessary as due to delay past commands will be executed in the future.

The action, a, is the delta term in the command law.

The reward method is the average measurements squared from the wavefront sensor.

#### RL References:
+ Soft Actor Critic: Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications." arXiv preprint arXiv:1812.05905 (2018).
+ Residual Policy Learning: Silver, Tom, et al. "Residual policy learning." arXiv preprint arXiv:1812.06298 (2018).

