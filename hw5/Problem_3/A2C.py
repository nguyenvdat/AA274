import os
import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt

# suppress deprecation warning for now
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# maximum number of training episodes
NUM_EPISODES = 90
# maximum number of steps per episode
# CartPole-V0 has a maximum of 200 steps per episodes
MAX_EP_STEPS = 200
# reward discount factor
GAMMA = .6
# once MAX_EPISODES or ctrl-c is pressed, number of test episodes to run
NUM_TEST_EPISODES = 3
# batch size used for the training
BATCH_SIZE = 1000
# maximum number of transitions stored in the replay buffer
MAX_REPLAY_BUFFER_SIZE = BATCH_SIZE * 10
# reward that is returned when the episode terminates early (i.e. the controller fails)
FAILURE_REWARD = -10.
# path where to save the actor after training
FROZEN_ACTOR_PATH = 'frozen_actor.pb'

# setting the random seed makes things reproducible
random_seed = 2
np.random.seed(random_seed)
tf.compat.v1.random.set_random_seed(random_seed)
tf.keras.backend.set_floatx('float32')


class Actor():
    def __init__(self, sess, state_dim, action_dim):
        """
        An actor for Actor-Critic reinforcement learning. This actor represent
        a stochastic policy. It predicts a distribution over actions condition
        on a given state. The distribution can then be sampled to produce
        an single control action.

        arguments:
            sess: a tensorflow session
            state_dim: an integer, number of states of the system
            action_dim: an integer, number of possible actions of the system
        returns:
            nothing
        """
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim

        # those placeholders serve as "inputs" to the computational graph.

        # state_input_ph is the input to the neural network
        self.state_input_ph = tf.compat.v1.placeholder(tf.float32, [None, state_dim], name='actor_state_input')
        # action_ph will be a label in the training process of the actor
        self.action_ph = tf.compat.v1.placeholder(tf.int32, [None, 1], name='actor_action')
        # td_error_ph will also be a label in the training process of the actor
        self.td_error_ph = tf.compat.v1.placeholder(tf.float32, [None, 1], name='actor_td_error')

        # setting up the computation graph

        # the neural network (input will be state, output is unscaled probability distribution)
        # note: the neural network must be entirely linear to support verification
        self.nn = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu',
                                  input_shape=[state_dim],
                                  kernel_initializer=tf.random_normal_initializer(0., .1),
                                  bias_initializer=tf.constant_initializer(.1),
                                  name='actor_h1'),
            tf.keras.layers.Dense(action_dim, activation=None,
                                  kernel_initializer=tf.random_normal_initializer(0., .1),
                                  bias_initializer=tf.constant_initializer(.1),
                                  name='actor_outputs'),
        ])
        # probability distribution over potential actions
        self.action_probs = tf.math.softmax(self.nn(self.state_input_ph))
        # convert action label to one_hot format
        self.action_one_hot = tf.one_hot(self.action_ph[:,0], self.action_dim, dtype='float32')
        # log of the action probability, cliped for numerical stability
        self.log_action_prob = tf.reduce_sum(tf.math.log(
            tf.clip_by_value(self.action_probs, 1e-14, 1.)) * self.action_one_hot, axis=1, keepdims=True)
        # the expected reward to go for this sample (J(theta)) (Eqn. 11)
        self.expected_v = self.log_action_prob * self.td_error_ph
        # taking the negative so that we effectively maximize
        self.loss = -tf.reduce_mean(self.expected_v)
        # the training step
        self.train_op = tf.compat.v1.train.AdamOptimizer(.01).minimize(self.loss)

    def train_step(self, state, action, td_error):
        """
        Runs the training step

        arguments:
            state: a tensor representing a batch of states (batch_size X state_dim)
            action: a tensor of integers representing a batch of actions (batch_size X 1)
            where the integers correspond to the action number (0 indexed)
            td_error: a tensor of floats (batch_size X 1) the temporal differences
        returns:
            expected_v: a tensor of the expected reward for each of the samples
            in the batch (batch_size X 1)
        """
        expected_v, _ = self.sess.run([self.expected_v, self.train_op],
                                      {self.state_input_ph: state,
                                       self.action_ph: action,
                                       self.td_error_ph: td_error})
        return expected_v

    def get_action(self, state):
        """
        Get an action for a given state by predicting a probability distribution
        over actions and sampling one.

        arguments:
            state: a tensor of size (state_dim) representing the current state
        returns:
            action: an integer (0 indexed) corresponding to the action taken by the actor
        """
        action_probs = self.sess.run(self.action_probs,
                                     {self.state_input_ph: state[np.newaxis, :]})
        action = np.random.choice(self.action_dim, p=action_probs[0, :])
        return action

    def export(self, frozen_actor_path):
        """
        Exports the neural network underlying the actor and its weights

        arguments:
            frozen_actor_path: a string, the path where to save the actor network
        returns:
            nothing
        """
        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(self.sess,
                                                                        tf.compat.v1.get_default_graph().as_graph_def(),
                                                                        ['sequential/actor_outputs/BiasAdd'])
        with tf.io.gfile.GFile (frozen_actor_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())


class Critic():
    def __init__(self, sess, state_dim):
        """
        A critic for Actor-Critic reinforcement learning. This critic works
        by estimating a value function (expected reward-to-go) for given
        states. It is trained using Temporal Difference error learning (TD error).

        arguments:
            sess: tensorflow session
            state_dim: an interger, number of states of the system
        returns:
            nothing
        """
        self.sess = sess
        self.state_dim = state_dim

        # those placeholders serve as "inputs" to the computational graph.

        # state_input_ph is the input to the neural network
        self.state_input_ph = tf.compat.v1.placeholder(tf.float32, [None, state_dim], name='critic_state_input')
        # reward_ph will be a label during the training process
        self.reward_ph = tf.compat.v1.placeholder(tf.float32, [None, 1], name='critic_reward_input')
        # v_next will be a 'label' during the training process, even though it is 
        # produced by the nerual network as well
        self.v_next_ph = tf.compat.v1.placeholder(tf.float32, [None, 1])

        # setting up the computation graph

        ######### Your code starts here #########
        # hint: look at the implementation of the actor, the TD error and
        # the loss functions described in the writeup. An neural network architecture
        # identical to the one used by the actor should do the trick, but feel free to
        # experiment!
        # Note that the current train_step and train_op code expect you to compute three
        # member variables: self.v (the reward-to-go), self.loss and self.td_error

        # the neural network (input will be the current state, output is an 
        # estimate of the reward-to-go)
        self.critic_nn = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu',
                                  input_shape=[state_dim],
                                  kernel_initializer=tf.random_normal_initializer(0., .1),
                                  bias_initializer=tf.constant_initializer(.1),
                                  name='critic_h1'),
            tf.keras.layers.Dense(1, activation=None,
                                  kernel_initializer=tf.random_normal_initializer(0., .1),
                                  bias_initializer=tf.constant_initializer(.1),
                                  name='critic_outputs'),
        ])
        self.v = self.critic_nn(self.state_input_ph)
        y = self.reward_ph + GAMMA*self.v_next_ph
        self.td_error = y - self.v
        self.loss = tf.math.reduce_mean(tf.math.multiply(self.td_error, self.td_error))
        ######### Your code ends here #########

        # the train step
        self.train_op = tf.compat.v1.train.AdamOptimizer(.01).minimize(self.loss)

    def train_step(self, state, reward, state_next):
        """
        Runs the training step

        arguments:
            state: a tensor representing a batch of initial states (batch_size X state_dim)
            reward: a tensor representing a batch of rewards (batch_size X 1)
            state_next: a tensor representing a batch of 'future states' (batch_size X state_dim)
            each sample (state, reward, state_next) correspond to a given transition
        returns:
            td_error: the td errors of the batch, as a numpy array (batch_size X 1)
        """
        v_next = self.sess.run(self.v, {self.state_input_ph: state_next})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.state_input_ph: state,
                                     self.reward_ph: reward,
                                     self.v_next_ph: v_next})
        return td_error


def run_actor(env, actor, num_episodes, render=True):
    """
    Runs the actor on the environment for
    num_episodes

    arguments:
        env: the openai gym environment
        actor: an instance of the Actor class
        num_episodes: number of episodes to run the actor for
    returns:
        nothing
    """
    for i_episode in range(num_episodes):
        state = env.reset()
        total_reward = 0.
        for t in range(MAX_EP_STEPS):
            if render:
                env.render()
            action = actor.get_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                print("Reward: ", str(total_reward))
                break


def train_actor_critic(sess):
    # setup the OpenAI gym environment
    env = gym.make('CartPole-v0')
    env.seed(random_seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # create an actor and a critic network and initialize their variables
    actor = Actor(sess, state_dim, action_dim)
    critic = Critic(sess, state_dim)
    sess.run(tf.compat.v1.global_variables_initializer())

    # the replay buffer will store observed transitions
    replay_buffer = np.zeros((0, 2*state_dim + 2))

    # you can stop the training at any time using ctrl+c (the actor will
    # still be tested and its network exported for verification

    # allocate memory to keep track of episode rewards
    reward_history = np.zeros(NUM_EPISODES)

    try:
        for i_episode in range(NUM_EPISODES):

            # very inneficient way of making sure the buffer isn't too full
            if replay_buffer.shape[0] > MAX_REPLAY_BUFFER_SIZE:
                replay_buffer = replay_buffer[-MAX_REPLAY_BUFFER_SIZE:, :]
            
            # reset the OpenAI gym environment to a random initial state for each episode
            state = env.reset()
            episode_reward = 0.

            for t in range(MAX_EP_STEPS):
                # uses the actor to get an action at the current state
                action = actor.get_action(state)
                # call gym to get the next state and reward, given we are taking action at the current state
                state_next, reward, done, info = env.step(action)
                # done=True means either the cartpole failed OR we've reached the maximum number of episode steps
                if done and t < (MAX_EP_STEPS - 1):
                    reward = FAILURE_REWARD
                # accumulate the reward for this whole episode
                episode_reward += reward
                # store the observed transition in our replay buffer for training
                replay_buffer = np.vstack((replay_buffer, np.hstack((state, action, reward, state_next))))

                # if our replay buffer has accumulated enough samples, we start learning the actor and the critic
                if replay_buffer.shape[0] >= BATCH_SIZE:

                    # we sample BATCH_SIZE transition from our replay buffer
                    samples_i = np.random.choice(replay_buffer.shape[0], BATCH_SIZE, replace=False)
                    state_samples = replay_buffer[samples_i, 0:state_dim]
                    action_samples = replay_buffer[samples_i, state_dim:state_dim+1]
                    reward_samples = replay_buffer[samples_i, state_dim+1:state_dim+2]
                    state_next_samples = replay_buffer[samples_i, state_dim+2:2*state_dim+2]

                    # compute the TD error using the critic
                    td_error = critic.train_step(state_samples, reward_samples, state_next_samples)

                    # train the actor (we don't need the expected value unless you want to log it)
                    actor.train_step(state_samples, action_samples, td_error)

                    if done:
                        # print how well we did on this episode
                        print(episode_reward)
                        reward_history[i_episode] = episode_reward
                
                # update current state for next iteration
                state = state_next
                
                if done:
                    break
            reward_history[i_episode] = episode_reward

    except KeyboardInterrupt:
        print("training interrupted")

    # plot reward history
    plt.figure()
    plt.plot(reward_history)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Episode Reward')
    plt.title('History of Episode Reward')
    if not os.path.exists('../plots'):
        os.makedirs('../plots')
    plt.savefig('../plots/p2_reward_history.png')
    plt.show()
    run_actor(env, actor, NUM_TEST_EPISODES)

    # exports the actor neural network and its weights, for future verification
    actor.export(FROZEN_ACTOR_PATH)
    # closes the environement
    env.close()


if __name__ == "__main__":
    sess = tf.compat.v1.Session()
    train_actor_critic(sess)
