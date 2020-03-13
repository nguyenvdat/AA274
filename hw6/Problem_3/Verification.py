import numpy as np
import copy
import tensorflow as tf
from maraboupy import Marabou
from maraboupy.MarabouUtils import addInequality

tf.keras.backend.set_floatx('float32')


def load_actor_network(frozen_actor_path):
    """
    loads the actor network from saved file

    arguments:
        frozen_actor_path: a string with the path of the frozen network
    returns:
        network: the network loaded in Marabou
        network_inputs: a list of variable indices which correspond to the neural network inputs
        network_outputs: a list of variable indices which correspond to the neural network outputs
    """
    network = Marabou.read_tf(frozen_actor_path,
                              savedModel=False)
                            #   inputName=['actor_state_input'])
                            #   outputName='sequential/actor_outputs/BiasAdd')
    network_inputs = network.inputVars[0][0]
    network_outputs = network.outputVars[0]
    return(network, network_inputs, network_outputs)


def check_actor_correct(network, network_inputs, network_outputs,
                        state_low, state_high, correct_action):
    """
    loads the actor network from saved file

    arguments:
        network, network_inputs, network_outputs: see load_actor_network
        state_low: a list of float, the low limit of the input state bounding box
        state_high: a list of float, the high limit of the input state bounding box
        correct_action: a zero-index integer corresponding to the CORRECT action
        that should be taken by the actor network for this input range
    returns:
        a boolean that is TRUE if the network always returns the correct
        input, FALSE otherswise. By correct, we mean correct_action is taken
        with a higher probability

    positive pole angle is falling towards the right,
    so if the pole angle is positive, and the actor network's
    first output is larger than the second output,
    then the pole is falling towards the right, and the most likely action is
    to push the cart to the left, which is WRONG
    if the result is UNSAT, this is GOOD, because the actor is therefore
    never wrong for the given range of states

    Hint: You will want to use some of the following functions:
    -- network.setUpperBound (x,v)
        arguments:
            x: (int) variable number to set
            v: (float) value representing upper bound

    --network.setLowerBound(x,v)
        arguments:
            x: (int) variable number to set
            v: (float) value representing lower bound

    --addInequality(network, vars, coeffs, scalar) (FYI)
        Function to conveniently add inequality constraint to network
        arguments:
            network: (MarabouNetwork) to which to add constraint
            vars: (list) of variable numbers
            coeffs: (list) of coefficients
            scalar: (float) representing RHS of equation
    """
    print(network_inputs)
    for i in range(len(state_low)):
        ######### Your code starts here #########
        # Set your lower and upper bound using
        # network.setLowerBound() and network.setUpperBound()
        network.setLowerBound(network_inputs[i], state_low[i])
        network.setUpperBound(network_inputs[i], state_high[i])
        ######### Your code ends here #########
    if correct_action == 0:
        addInequality(network,
                      [network_outputs[0], network_outputs[1]],
                      [1., -1.], 0.)
    else:
        addInequality(network,
                      [network_outputs[0], network_outputs[1]],
                      [-1., 1.], 0.)
    vals, stats = network.solve()
    if len(vals) > 0:
        # SAT
        return False
    # UNSAT
    return True


if __name__ == "__main__":
    (net, net_inputs, net_outputs) = load_actor_network(
        './mdl/frozen_actor.pb')

    # CartPole states are
    # [Cart Position, Cart Velocity, Pole Angle, Pole Velocity]
    # CartPole actions are
    # Action 0 -> Push cart to the left
    # Action 1 -> Push cart to the right
    state_low = [0., 0., .01, 0.]
    state_high = [0., 0., .2, 0.]
    correct_action = 1

    if check_actor_correct(net, net_inputs, net_outputs,
                           state_low, state_high, correct_action):
        print('The query is UNSAT')
        print('The network cannot produce incorrect' +
              ' probabilities for this state range')
    else:
        print('The query is SAT')
        print('The network can produce incorrect' +
              ' probabilities for this state range')
