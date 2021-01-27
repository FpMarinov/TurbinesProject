The Project is fully described in [Dissertation.pdf](Dissertation.pdf)

The ability to predict the thrust or torque applied to marine turbines by using upstream flow velocity and torque or thrust respectively could improve the lifespan and function of the turbines. This project attempts to develop software, which is capable of such predictions. The approach is split in two stages. The first stage is to use Variational Autoencoders made up of convolutional and fully connected layers to create meaningful latent space representations of the individual parameters (flow velocity, thrust and torque). In the second stage the representations are used in combinations of two to try and predict the third parameter with a decoder neural network. Training for the first stage observes sensible learning over 100 epochs after the units of velocity and thrust are modified so that all parameters' standard deviations have the same order of magnitude as that of torque, O(1), which was the best performing parameter prior to the unit change. For the second stage the results vary. The results when trying to predict torque and thrust with 100 epochs of training are sensible, but velocity prediction was not achieved. The difficulties in predicting velocity could possibly be due to the fact that the standard deviation for velocity is 3 times smaller than those for thrust and torque after the unit change. Another possible explanation is the turbulence in the flow, which would be hard to account for with just two parameters. Future work could explore these ideas further in an attempt to produce a decoder capable of predicting flow velocity from thrust and torque.
