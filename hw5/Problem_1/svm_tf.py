import numpy as np
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf1
import tensorflow.compat.v2 as tf
from utils import *
from HOG import *

tf1.compat.v1.enable_eager_execution()


def identity_phi(x):
    return x


def circle_phi(x):
    ######### Your code starts here #########

    ######### Your code ends here #########
    return x


def inner_circle_phi(x):
    ######### Your code starts here #########
    
    ######### Your code ends here #########
    return x


class SVM(tf.keras.Model):
    def __init__(self, dim):
        super(SVM, self).__init__()
        self.W = tf.Variable(tf.random.normal([dim, 1]), name="weights")
        self.b = tf.Variable(tf.zeros([1,]), name="weights")

    def call(self, x, is_prediction=True):
        """
        This Function calculates y_est = xW-b during training.
        :param is_prediction: Boolean, describing if we train/evaluate the model or predict using the trained model.
        :param x: Input data points
        :return: Tensor y_est with shape(dim,1)
        Hint: use tf native functions to perform operations!
        Hint: your return values should be different based on is_prediction.
        """
        ######### Your code starts here #########
        





        ######### Your code ends here #########


def loss(y_est, y, W, lam):
    """
    Returns soft-margin hinge SVM loss. 
    """
    y = tf.cast(y, dtype=tf.float32)
    ######### Your code starts here #########
    
    ######### Your code ends here #########
    return loss


def svm(data, basis_function=identity_phi, epochs=10, ret_dec_p=False):
    """
    Trains and tests SVM classifier. 
    """
    params = {
        'train_batch_size': 32,
        'eval_batch_size': 32,
        ######### Your code starts here #########
        # define your learning rate ('lr') and lambda ('lam') values here
        

        ######### Your code ends here #########
    }
    dim = basis_function(data['x_train']).shape[-1]
    svm_model = SVM(dim=dim)
    optimizer = tf.keras.optimizers.SGD(learning_rate=params['lr'])

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')
    eval_loss = tf.keras.metrics.Mean(name='eval_loss')
    eval_accuracy = tf.keras.metrics.Accuracy(name='eval_accuracy')

    @tf.function
    def train_step(x, y):
        ######### Your code starts here #########
        # We want to perform a single training step (for one batch):
        # 1. Make a forward pass through the model
        # 2. Calculate the loss for the output of the forward pass
        # 3. Based on the loss calculate the gradient for all weights
        # 4. Run an optimization step on the weights.
        # Helpful Functions: tf.GradientTape(), tf.GradientTape.gradient(), tf.keras.Optimizer.apply_gradients
        




        ######### Your code ends here #########

        train_loss(current_loss)
        train_accuracy(y, svm_model(x))

    @tf.function
    def train(train_data):
        for x, y in train_data:
            train_step(x, y)

    @tf.function
    def eval_step(x, y):
        ######### Your code starts here #########
        # We want to evaluate the model on a single batch:
        # 1. Make a forward pass through the model
        # 2. Calculate the loss for the output of the forward pass
        # 3. Get the predicted labels for the batch
        # 4. Add to the values to the respective metrics -> See train_step
        



        ######### Your code ends here #########

    @tf.function
    def eval(eval_data):
        for x, y in eval_data:
            eval_step(x, y)

    train_data = tf.data.Dataset.from_tensor_slices((basis_function(data['x_train']), data['y_train'])).shuffle(10000).batch(params['train_batch_size'])
    eval_data = tf.data.Dataset.from_tensor_slices((basis_function(data['x_eval']), data['y_eval'])).shuffle(10000).batch(params['eval_batch_size'])

    for epoch in range(epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        eval_loss.reset_states()
        eval_accuracy.reset_states()

        train(train_data)
        eval(eval_data)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              eval_loss.result(),
                              eval_accuracy.result() * 100))

    y_pred = svm_model(basis_function(data['x_pred'])).numpy().squeeze()

    if data['name'] == 'hog':
        svm_model.save_weights('./hog_model/trained_weights')

    if ret_dec_p:
        return y_pred, (-svm_model.W[0].numpy(), (svm_model.b / svm_model.W[-1]).numpy())
    return y_pred


def get_hog_data():
    """
    This function calls the hog_descriptor function to obtain hog features for a dataset.
    Luckily, we have implemented hog_descriptor for you.
    hog_descriptor uses tensorflow to compute the features, therefore, its outputs are obtained by running a graph
    Recall that in order to access values at a particular node of a graph, you must use
     .numpy() on the tensor of interest.
    """
    pedestrian_data = np.load("pedestrian_dataset.npz")
    ######### Your code starts here #########
    




    
    ######### Your code ends here #########
    return (x_train, y_train), (x_eval, y_eval), (x_pred, y_true)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--type', type=str, help="linear, non_linear, circle, inner_circle or hog", default="linear")
    parser.add_argument('--feature', type=str, default="identity", help="identity or custom")

    args = parser.parse_args()
    maybe_makedirs("../plots")
    dec_p = None
    basis_func_problems = ['circle', 'inner_circle', 'moons']
    if args.type == "linear":
        x_train, y_train = generate_data_lin(N=5000)
        x_eval, y_eval = generate_data_lin(N=1000)
        x_pred, y_true = generate_data_lin(N=1000)

        data = {}
        data['name'] = 'toy'
        data['x_train'] = x_train.astype('float32')
        data['y_train'] = y_train.astype('float32')
        data['x_eval'] = x_eval.astype('float32')
        data['y_eval'] = y_eval.astype('float32')
        data['x_pred'] = x_pred.astype('float32')
        data['y_true'] = y_true.astype('float32')

        predicted_labels, dec_p = svm(data, ret_dec_p=True, epochs=10)

        plt.figure(figsize=(10, 3))
        plt.subplot(1, 2, 1)
        c = ['powderblue' if lb == 1 else 'indianred' for lb in y_train]
        plt.scatter(x_train[:, 0], x_train[:, 1], c=c, alpha=0.5, s=50)
        plt.title("Dataset to be classified using %s features" % args.feature)

        plt.subplot(1, 2, 2)
        c = ['powderblue' if lb == 1 else 'indianred' for lb in predicted_labels]
        plt.scatter(x_pred[:, 0], x_pred[:, 1], c=c, s=50, alpha=0.5)
        # misclassified data
        d = predicted_labels - y_true[:, 0]
        misclass_idx = np.where(d != 0)[0]
        c = ['red' if lb == 2 else 'blue' for lb in d[misclass_idx]]
        plt.scatter(x_pred[misclass_idx, 0], x_pred[misclass_idx, 1], c=c, s=50, alpha=0.8)

        if dec_p is not None:
            p1, p2 = dec_p
            x1_dec = np.arange(-1.5, 1.5, 0.2)
            x2_dec = p1 * x1_dec + p2
            plt.plot(x1_dec, x2_dec, c='green')

        accuracy = 100 * (1 - len(misclass_idx) / float(x_pred.shape[0]))
        plt.title("Classification results: %.2f%%" % accuracy)
        plt.savefig('../plots/svm_linear.png')
        plt.show()

    elif args.type == "non_linear":
        x_train, y_train = generate_data_non_lin(N=5000)
        x_eval, y_eval = generate_data_non_lin(N=1000)
        x_pred, y_true = generate_data_non_lin(N=1000)
        data = {}
        data['name'] = 'toy'
        data['x_train'] = x_train.astype('float32')
        data['y_train'] = y_train.astype('float32')
        data['x_eval'] = x_eval.astype('float32')
        data['y_eval'] = y_eval.astype('float32')
        data['x_pred'] = x_pred.astype('float32')
        data['y_true'] = y_true.astype('float32')

        predicted_labels, dec_p = svm(data, ret_dec_p=True, epochs=20)

        plt.figure(figsize=(24, 8))
        plt.subplot(1,2,1)
        c = ['powderblue' if lb == 1 else 'indianred' for lb in y_train]
        plt.scatter(x_train[:,0], x_train[:,1], c = c, alpha=0.5, s=50)
        plt.legend(handles = [plt.scatter([],[],c='powderblue',s=50,alpha=0.5), plt.scatter([],[],c='indianred',s=50,alpha=0.5)],
        		   labels=['Class A', 'Class B'])
        plt.title("Dataset to be classified using %s features" % args.feature)

        plt.subplot(1,2,2)
        c = ['powderblue' if lb == 1 else 'indianred' for lb in predicted_labels]
        plt.scatter(x_pred[:,0], x_pred[:,1], c=c, s=50, alpha=0.5)
        # misclassified data
        d = predicted_labels - y_true[:,0]
        misclass_idx = np.where(d!= 0)[0]
        c = ['red' if lb == 2 else 'blue' for lb in d[misclass_idx]]
        plt.scatter(x_pred[misclass_idx,0], x_pred[misclass_idx,1], c=c, s=50, alpha=0.8)
        plt.legend(handles = [plt.scatter([],[],c='powderblue',s=50,alpha=0.5),
        					  plt.scatter([],[],c='indianred',s=50,alpha=0.5),
        					  plt.scatter([],[],c='red',s=50,alpha=0.8),
        					  plt.scatter([],[],c='blue',s=50,alpha=0.8)],
           			labels=['Class A', 'Class B', 'Misclassified Class B', 'Misclassified Class A'])

        if dec_p is not None:
            p1, p2 = dec_p
            x1_dec = np.arange(-3, 3, 0.2)
            x2_dec = p1 * x1_dec + p2
            plt.plot(x1_dec, x2_dec, c='green')

        accuracy = 100*(1-len(misclass_idx)/float(x_pred.shape[0]))
        plt.title("Classification results: %.2f%%" % accuracy )
        plt.savefig('../plots/svm_non_linear.png')
        plt.show()

    elif args.type in basis_func_problems:
        x_train, y_train = generate_data_basis(args.type, N=5000)
        x_eval, y_eval = generate_data_basis(args.type, N=1000)
        x_pred, y_true = generate_data_basis(args.type, N=1000)

        data = {}
        data['name'] = 'toy'
        data['x_train'] = x_train.astype('float32')
        data['y_train'] = y_train.astype('float32')
        data['x_eval'] = x_eval.astype('float32')
        data['y_eval'] = y_eval.astype('float32')
        data['x_pred'] = x_pred.astype('float32')
        data['y_true'] = y_true.astype('float32')

        basis_function = globals()[args.type + "_phi"] if args.feature == "custom" else identity_phi
        predicted_labels = svm(data, basis_function=basis_function)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        c = ['powderblue' if lb == 1 else 'indianred' for lb in y_train]
        plt.scatter(x_train[:, 0], x_train[:, 1], c=c, alpha=0.5, s=50)
        plt.title("Dataset to be classified using %s features" % args.feature)

        plt.subplot(1, 2, 2)
        c = ['powderblue' if lb == 1 else 'indianred' for lb in predicted_labels]
        plt.scatter(x_pred[:, 0], x_pred[:, 1], c=c, s=50, alpha=0.5)
        # misclassified data
        d = predicted_labels - y_true[:, 0]
        misclass_idx = np.where(d != 0)[0]
        c = ['red' if lb == 2 else 'blue' for lb in d[misclass_idx]]
        plt.scatter(x_pred[misclass_idx, 0], x_pred[misclass_idx, 1], c=c, s=50, alpha=0.8)
        accuracy = 100 * (1 - len(misclass_idx) / float(x_pred.shape[0]))
        plt.title("Classification results: %.2f%%" % accuracy)

        plt.savefig("../plots/svm_basis_%s_%s.png" % (args.type, args.feature))
        plt.show()

    elif args.type == "hog":
        (x_train, y_train), (x_eval, y_eval), (x_pred, y_true) = get_hog_data()
        data = {}
        data['name'] = 'hog'
        data['x_train'] = x_train
        data['y_train'] = y_train
        data['x_eval'] = x_eval
        data['y_eval'] = y_eval
        data['x_pred'] = x_pred
        data['y_true'] = y_true
        predicted_labels = svm(data, epochs=600)
        d = predicted_labels - y_true[:, 0]
        misclass_idx = np.where(d != 0)[0]
        np.save("hog_misclass_idx.npy", misclass_idx)
        accuracy = 100 * (1 - len(misclass_idx) / float(x_pred.shape[0]))
        print("\n\n\n\nClassification results: %.2f%%\n\n\n" % accuracy)