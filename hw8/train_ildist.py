import numpy as np
import tensorflow as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
import argparse
from utils import *

tf1.compat.v1.enable_eager_execution()

class NN(tf.keras.Model):
    def __init__(self, in_size, out_size):
        super(NN, self).__init__()
        
        ######### Your code starts here #########
        # We want to define and initialize the weights & biases of the neural network.
        # - in_size is dim(O)
        # - out_size is dim(A) = 2
        # IMPORTANT: out_size is still 2 in this case, because the action space is 2-dimensional. But your network will output some other size as it is outputing a distribution!
        # HINT 1: An example of this was given to you in Homework 1's Problem 1 in svm_tf.py. Now you will implement a multi-layer version.
        # HINT 2: You should use either of the following for weight initialization:
        #           - tf1.contrib.layers.xavier_initializer (this is what we tried)
        #           - tf.keras.initializers.GlorotUniform (supposedly equivalent to the previous one)
        #           - tf.keras.initializers.GlorotNormal
        #           - tf.keras.initializers.he_uniform or tf.keras.initializers.he_normal
        self.dense1 = tf.keras.layers.Dense(50, kernel_initializer=tf.keras.initializers.glorot_normal(), name='dense_1')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.av1 = tf.keras.layers.Activation('relu')
        self.drop1 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(50, kernel_initializer=tf.keras.initializers.glorot_normal(), name='dense_2')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.av2 = tf.keras.layers.Activation('relu')
        self.drop2 = tf.keras.layers.Dropout(0.2)
        self.dense3 = tf.keras.layers.Dense(out_size*3-1, kernel_initializer=tf.keras.initializers.glorot_normal(), name='dense_3')
        self.av3 = tf.keras.layers.Activation('relu')
        ########## Your code ends here ##########

    def call(self, x):
        x = tf.cast(x, dtype=tf.float32)
        ######### Your code starts here #########
        # We want to perform a forward-pass of the network. Using the weights and biases, this function should give the network output for x where:
        # x is a (? x |O|) tensor that keeps a batch of observations
        # IMPORTANT: First two columns of the output tensor must correspond to the mean vector!
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.av1(x)
        x = self.drop1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.av2(x)
        x = self.drop2(x)
        x = self.dense3(x)
        # x = self.av3(x)
        # x[:,2:]
        return tf.concat([x[:,:2], self.av3(x[:,2:])+1e-6], axis=1)
        ########## Your code ends here ##########


   
def loss(y_est, y):
    y = tf.cast(y, dtype=tf.float32)
    ######### Your code starts here #########
    # We want to compute the negative log-likelihood loss between y_est and y where
    # - y_est is the output of the network for a batch of observations,
    # - y is the actions the expert took for the corresponding batch of observations
    # At the end your code should return the scalar loss value.
    # HINT: You may find the classes of tensorflow_probability.distributions (imported as tfd) useful.
    #       In particular, we used MultivariateNormalTriL, but it is not the only way.
    mvn = tfd.MultivariateNormalTriL(loc=y_est[:,:2], scale_tril=tfp.math.fill_triangular(y_est[:,2:]))
    return -tf.math.reduce_mean(mvn.log_prob(y))
    ########## Your code ends here ##########


def nn(data, args):
    """
    Trains a feedforward NN. 
    """
    params = {
        'train_batch_size': 4096*32,
    }
    in_size = data['x_train'].shape[-1]
    out_size = data['y_train'].shape[-1]
    
    nn_model = NN(in_size, out_size)
    if args.restore:
        nn_model.load_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_ILDIST')
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def train_step(x, y):
        ######### Your code starts here #########
        # We want to perform a single training step (for one batch):
        # 1. Make a forward pass through the model
        # 2. Calculate the loss for the output of the forward pass
        # 3. Based on the loss calculate the gradient for all weights
        # 4. Run an optimization step on the weights.
        # Helpful Functions: tf.GradientTape(), tf.GradientTape.gradient(), tf.keras.Optimizer.apply_gradients
        # HINT: You did the exact same thing in Homework 1! It is just the networks weights and biases that are different.
        with tf.GradientTape() as t:
            y_est = nn_model(x)
            current_loss = loss(y_est, y)
        grad = t.gradient(current_loss, nn_model.trainable_variables)
        optimizer.apply_gradients(zip(grad, nn_model.trainable_variables))
        ########## Your code ends here ##########

        train_loss(current_loss)

    @tf.function
    def train(train_data):
        for x, y in train_data:
            train_step(x, y)


    train_data = tf.data.Dataset.from_tensor_slices((data['x_train'], data['y_train'])).shuffle(100000).batch(params['train_batch_size'])

    for epoch in range(args.epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()

        train(train_data)

        template = 'Epoch {}, Loss: {}'
        print(template.format(epoch + 1, train_loss.result()))
    nn_model.save_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_ILDIST')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--goal', type=str, help="left, straight, right, inner, outer, all", default="all")
    parser.add_argument('--scenario', type=str, help="intersection, circularroad", default="intersection")
    parser.add_argument("--epochs", type=int, help="number of epochs for training", default=1000)
    parser.add_argument("--lr", type=float, help="learning rate for Adam optimizer", default=1e-3)
    parser.add_argument("--restore", action="store_true", default=False)
    args = parser.parse_args()
    
    maybe_makedirs("./policies")
    
    data = load_data(args)

    nn(data, args)
