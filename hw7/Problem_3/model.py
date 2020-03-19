import tensorflow as tf1
import tensorflow.compat.v2 as tf
tf1.compat.v1.enable_eager_execution()

DIM_IMG = (224, 224)

class AccelerationLaw(tf.keras.layers.Layer):
    """
    Tensorflow layer to evaluate the acceleration law:

        a = g * (sin(th) - mu * cos(th))

    g is a trainable parameter because the units of acceleration in the
    dataset are pixels/frame^2, and the conversion from 9.81 m/s^2 to these
    units are unknown.
    """

    def __init__(self, **kwargs):
        super(AccelerationLaw, self).__init__(**kwargs)

    def build(self, input_shape):
        self.g = self.add_weight(name='g', shape=(1,), initializer=tf.keras.initializers.Constant(16), trainable=True)

    def call(self, inputs):
        mu, th = inputs

        ########## Your code starts here ##########
        a = self.g * (tf.math.sin(th) - mu*tf.math.cos(th))  # TODO
        ########## Your code ends here ##########

        # Ensure output acceleration is positive
        return a

def build_model():
    """
    Build the acceleration prediction network.

    The network takes two inputs:
        img - first frame of the video
        th  - incline angle of the ramp [rad]

    The output is:
        a - predicted acceleration of the object [pixels/frame^2]

    The last two layers of the network before the AccelerationLaw layer should be:
        p_class - A fully connected layer of size 32 with softmax output. This
                  represents a probability distribution over 32 possible classes
                  for the material of the object.
                  NOTE: Name this layer 'p_class'!
        mu - A vector of 32 weights representing the friction coefficients of
             each material class. The dot product of these weights and p_class
             represent the predicted friction coefficient of the object in the
             video.
             NOTE: Name this layer 'mu'!
    """

    img_input = tf.keras.Input(shape=(DIM_IMG[1], DIM_IMG[0], 3), name='img')
    th_input = tf.keras.Input(shape=(1,), name='th')

    ########## Your code starts here ##########
    # TODO: Create your neural network and replace the following two layers
    #       according to the given specification.
    conv1 = tf.keras.layers.Conv2D(20, (3,3), strides=(2,2), padding='same', name='conv1')(img_input)
    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    av1 = tf.keras.layers.Activation('relu')(bn1)
    drop1 = tf.keras.layers.Dropout(0.2)(av1)
    conv2 = tf.keras.layers.Conv2D(20, (3,3), strides=(2,2), padding='same')(drop1)
    bn2 = tf.keras.layers.BatchNormalization()(conv2)
    av2 = tf.keras.layers.Activation('relu')(bn2)
    drop2 = tf.keras.layers.Dropout(0.2)(av2)
    flatten = tf.keras.layers.Flatten()(drop2)
    p_class = tf.keras.layers.Dense(32)(flatten)
    p_class = tf.keras.layers.Softmax(name='p_class')(p_class)
    mu = tf.keras.layers.Dense(32, name='mu')(p_class)
    mu_pred = tf.math.reduce_sum(tf.math.multiply(p_class, mu), axis=-1, keepdims=True)
    ########## Your code ends here ##########
    a_pred = AccelerationLaw(name='a')((mu_pred, th_input))
    return tf.keras.Model(inputs=[img_input, th_input], outputs=[a_pred])

def build_baseline_model():
    """
    Build a baseline acceleration prediction network.

    The network takes one input:
        img - first frame of the video

    The output is:
        a - predicted acceleration of the object [pixels/frame^2]

    The structure of this network should match the other model before the
    p_class layer. Instead of outputting p_class, it should directly output a
    scalar value representing the predicted acceleration (without using the
    AccelerationLaw layer).
    """

    img_input = tf.keras.Input(shape=(DIM_IMG[1], DIM_IMG[0], 3), name='img')
    th_input = tf.keras.Input(shape=(1,), name='th')

    ########## Your code starts here ##########
    # TODO: Replace the following with your model from build_model().
    conv1 = tf.keras.layers.Conv2D(20, (3,3), strides=(2,2), padding='same', name='conv1')(img_input)
    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    av1 = tf.keras.layers.Activation('relu')(bn1)
    drop1 = tf.keras.layers.Dropout(0.2)(av1)
    conv2 = tf.keras.layers.Conv2D(20, (3,3), strides=(2,2), padding='same')(drop1)
    bn2 = tf.keras.layers.BatchNormalization()(conv2)
    av2 = tf.keras.layers.Activation('relu')(bn2)
    drop2 = tf.keras.layers.Dropout(0.2)(av2)
    flatten = tf.keras.layers.Flatten()(drop2)
    a_pred = tf.keras.layers.Dense(1, name='a')(flatten)
    ########## Your code ends here ##########

    return tf.keras.Model(inputs=[img_input], outputs=[a_pred])

def loss(a_actual, a_pred):
    """
    Loss function: L2 norm of the error between a_actual and a_pred.
    """

    ########## Your code starts here ##########
    error = a_actual - a_pred
    l = tf.math.reduce_mean(tf.math.multiply(error, error))  # TODO
    ########## Your code ends here ##########

    return l
