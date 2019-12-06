import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import os


def load_data(mode='train', max_n=None):
    """
    Function to (download and) load the MNIST data
    :param mode: train or test
    :return: data and the corresponding labels
    """
    print(f"\nMode: {mode}")
    if mode == 'train':
        raw_train = np.genfromtxt('CDM_leave_out_training_01.csv', delimiter=',')
        raw_validation = np.genfromtxt('CDM_leave_out_validation_01.csv', delimiter=',')
        
        data_train = raw_train[:max_n,:-1] if max_n else raw_train[:,:-1]
        data_validation = raw_validation[:max_n,:-1] if max_n else raw_validation[:,:-1]
        
        data_train_labels = raw_train[:max_n,-1] if max_n else raw_train[:,-1]
        data_validation_labels = raw_validation[:max_n,-1] if max_n else raw_validation[:,-1]
        
        x_train, y_train, x_valid, y_valid = data_train, data_train_labels, \
                                             data_validation, data_validation_labels
        return x_train, y_train, x_valid, y_valid
    elif mode == 'test':
        raw_test = np.genfromtxt('CDM_leave_out_test_01.csv', delimiter=',')
        data_test = raw_test[:max_n,:-1] if max_n else raw_test[:,:-1]
        data_test_labels = raw_test[:max_n,-1] if max_n else raw_test[:,-1]
        
        x_test, y_test = data_test, data_test_labels
        return x_test, y_test




if __name__ == "__main__":
    tf.disable_v2_behavior()
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Load data
    x_train, y_train, x_valid, y_valid = load_data(mode='train', max_n=1000000)
    print("Size of:")
    print("- Training-set:\t{}".format(len(y_train)))
    print("- Validation-set:\t{}".format(len(y_valid)))
    
    print('x_train:\t{}'.format(x_train.shape))
    print('y_train:\t{}'.format(y_train.shape))
    print('x_train:\t{}'.format(x_valid.shape))
    print('y_valid:\t{}'.format(y_valid.shape))
    
    n_components = x_train.shape[1]
    # Placeholder
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_components])
    Y = tf.placeholder(dtype=tf.float32, shape=[None])
    
    # Initializers
    sigma = 1
    weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
    bias_initializer = tf.zeros_initializer()
    
    # Model architecture parameters
    n_neurons_1 = 1024
    n_neurons_2 = 512
    n_neurons_3 = 256
    n_neurons_4 = 128
    n_target = 1
    # Layer 1: Variables for hidden weights and biases
    W_hidden_1 = tf.Variable(weight_initializer([n_components, n_neurons_1]))
    bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
    # Layer 2: Variables for hidden weights and biases
    W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
    bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
    # Layer 3: Variables for hidden weights and biases
    W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
    bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
    # Layer 4: Variables for hidden weights and biases
    W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
    bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

    # Output layer: Variables for output weights and biases
    W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
    bias_out = tf.Variable(bias_initializer([n_target]))
    
    # Hidden layer
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
    hidden_2 = tf.nn.sigmoid(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
    hidden_3 = tf.nn.sigmoid(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
    hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

    # Output layer (must be transposed)
    out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))
    
    # Cost function
    mse = tf.reduce_mean(tf.math.squared_difference(out, Y))
    
    # Optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)
    
    # Make Session
    net = tf.Session()
    # Run initializer
    net.run(tf.global_variables_initializer())

    # Number of epochs and batch size
    epochs = 10
    batch_size = 2000

    # Setup plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    line1, = ax1.plot(y_valid, y_valid, 'b.')
    plt.xlabel(f'Predicted Value')
    plt.ylabel(f'True Value')

    mse_values = list()
    for e in trange(epochs):
        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        x_train = x_train[shuffle_indices]
        y_train = y_train[shuffle_indices]

        # Minibatch training
        for i in trange(0, len(y_train) // batch_size):
            start = i * batch_size
            batch_x = x_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]
            # Run optimizer with batch
            net.run(opt, feed_dict={X: batch_x, Y: batch_y})
            mse_value = net.run(mse, feed_dict={X: x_valid, Y: y_valid})
            mse_values.append(mse_value)
            # Show progress
            if np.mod(i, 5) == 0:
                # Prediction
                pred = net.run(out, feed_dict={X: x_valid})
                line1.set_xdata(pred)
                plt.title(f'Epoch {e}, Batch {i}')
                file_name = f'img/epoch_{e}_batch_{i}.jpg'
                plt.savefig(file_name)
                plt.pause(0.01)
                
                
                
    # Print final MSE after Training
    mse_final = net.run(mse, feed_dict={X: x_valid, Y: y_valid})
    print(mse_final)
    
    plt.show()
    plt.figure()    
    plt.plot(range(len(mse_values)), mse_values, 'r')
    plt.title(f'Mean Squared Error')
    plt.xlabel(f'Iteration')
    plt.ylabel(f'MSE')
    plt.show()