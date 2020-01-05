"""Create a HyperparameterConfig for a TensorFlow ADAM optimizer.

"""
import metatree as mt


def adam_optimizer() -> mt.HyperparameterConfig:
    """Create a HyperparameterConfig for a TensorFlow ADAM optimizer.

    Returns:
        (mt.HyperparameterConfig): HyperparameterConfig for a TensorFlow ADAM
            optimizer.

    """
    # Create a dict of hyperparameters and their feasible ranges
    hyperparam_dict = {
        # ADAM optimization hyperparameters
        # Ref: Kingma, Diederik P., and Jimmy Ba.
        #      "Adam: A method for stochastic optimization." (2014).
        # log10 of alpha1 := 1 - beta1
        'log_alpha1': -1.5,
        # log10 of alpha2 := 1 - beta2
        'log_alpha2': -2.1,
        # log10 of epsilon
        'log_epsilon': -7.,
        # Other hyperparameters
        # Minimum value for the voxel weights in the summed xentropy op
        'weight_floor': 0.01,

        # Exponential learning rate decay hyperparams
        # log10 of the learning rate
        'log_learning_rate': -3.,
        # log10 of the number of Decay steps
        'log_decay_steps': 3.4,
        # Exponential decay rate
        'exponential_decay_rate': 0.75
    }

    return mt.HyperparameterConfig(hyperparam_dict)
