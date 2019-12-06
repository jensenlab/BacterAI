import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def generate_training_data():
    m = model.load_cobra("models/iSMUv01_CDM_LOO_v2.xml")
    max_n = 10000
    with open('CDM_leave_out_validation_01.csv', mode='a') as file:
        writer = csv.writer(file, delimiter=',')
        for _ in trange(max_n):
            # n = random.randint(0, len(model.KO_RXN_IDS))
            n = sp.poisson.rvs(5)
            grow, reactions = model.knockout_and_simulate(
                m, n, return_boolean=True)
            reactions = list(reactions)
            reactions.append(grow)
            writer.writerow(reactions)
            # print(grow, reactions)
        
        for _ in trange(max_n):
            n = random.randint(0, len(model.KO_RXN_IDS))
            grow, reactions = model.knockout_and_simulate(
                m, n, return_boolean=True)                
            reactions = list(reactions)
            reactions.append(grow)
            writer.writerow(reactions)
            # print(grow, reactions)
            
            
def load_data(mode='train', max_n=None):
    """
    Function to (download and) load the MNIST data
    :param mode: train or test
    :return: data and the corresponding labels
    """
    print(f"\nMode: {mode}")
    if mode == 'train':
        raw_train = np.genfromtxt('CDM_leave_out_training.csv', delimiter=',')
        raw_validation = np.genfromtxt('CDM_leave_out_validation.csv', delimiter=',')
        
        data_train = raw_train[:max_n,:-1] if max_n else raw_train[:,:-1]
        data_validation = raw_validation[:max_n,:-1] if max_n else raw_validation[:,:-1]
        
        data_train_labels = raw_train[:max_n,-1] if max_n else raw_train[:,-1]
        data_validation_labels = raw_validation[:max_n,-1] if max_n else raw_validation[:,-1]
        
        x_train, y_train, x_valid, y_valid = data_train, data_train_labels, \
                                             data_validation, data_validation_labels
        return x_train, y_train, x_valid, y_valid
    elif mode == 'test':
        raw_test = np.genfromtxt('CDM_leave_out_test.csv', delimiter=',')
        data_test = raw_test[:max_n,:-1] if max_n else raw_test[:,:-1]
        data_test_labels = raw_test[:max_n,-1] if max_n else raw_test[:,-1]
        
        x_test, y_test = data_test, data_test_labels
        return x_test, y_test

def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y

def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch

# weight and bais wrappers
def weight_variable(name, shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W_' + name,
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)

def bias_variable(name, shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name,
                           dtype=tf.float32,
                           initializer=initial)
    
def fc_layer(x, num_units, name, use_relu=True):
    """
    Create a fully-connected layer
    :param x: input from previous layer
    :param num_units: number of hidden units in the fully-connected layer
    :param name: layer name
    :param use_relu: boolean to add ReLU non-linearity (or not)
    :return: The output array
    """
    in_dim = x.get_shape()[1]
    W = weight_variable(name, shape=[in_dim, num_units])
    b = bias_variable(name, [num_units])
    layer = tf.matmul(x, W)
    layer += b
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


if __name__ == "__main__":
    # Load data
    x_train, y_train, x_valid, y_valid = load_data(mode='train', max_n=100)
    print("Size of:")
    print("- Training-set:\t{}".format(len(y_train)))
    print("- Validation-set:\t{}".format(len(y_valid)))
    
    print('x_train:\t{}'.format(x_train.shape))
    print('y_train:\t{}'.format(y_train.shape))
    print('x_train:\t{}'.format(x_valid.shape))
    print('y_valid:\t{}'.format(y_valid.shape))
    
    # Hyper-parameters
    epochs = 10             # Total number of training epochs
    batch_size = 100        # Training batch size
    display_freq = 100      # Frequency of displaying the training results
    learning_rate = 0.001   # The optimization initial learning rate
    
    # Network Config
    h1 = 200                # Number of units in the first hidden layer
    n_classes = 1
    # Create the graph for the linear model
    # Placeholders for inputs (x) and outputs(y)
    x = tf.placeholder(tf.float32, shape=[None, x_train.shape[1]], name='X')
    y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')
    
    # Create a fully-connected layer with h1 nodes as hidden layer
    fc1 = fc_layer(x, h1, 'FC1', use_relu=True)
    # Create a fully-connected layer with n_classes nodes as output layer
    output_logits = fc_layer(fc1, n_classes, 'OUT', use_relu=False)

    # Define the loss function, optimizer, and accuracy
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss')
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
    correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    
    # Network predictions
    cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')
    
    # Create the op for initializing all variables
    init = tf.compat.v1.global_variables_initializer()
    
    # Create an interactive session (to keep the session in the other cells)
    sess = tf.compat.v1.InteractiveSession
    # Initialize all variables
    sess.run(init)
    # Number of training iterations in each epoch
    num_tr_iter = int(len(y_train) / batch_size)
    for epoch in range(epochs):
        print('Training epoch: {}'.format(epoch + 1))
        # Randomly shuffle the training data at the beginning of each epoch 
        x_train, y_train = randomize(x_train, y_train)
        for iteration in range(num_tr_iter):
            start = iteration * batch_size
            end = (iteration + 1) * batch_size
            x_batch, y_batch = get_next_batch(x_train, y_train, start, end)

            # Run optimization op (backprop)
            feed_dict_batch = {x: x_batch, y: y_batch}
            sess.run(optimizer, feed_dict=feed_dict_batch)

            if iteration % display_freq == 0:
                # Calculate and display the batch loss and accuracy
                loss_batch, acc_batch = sess.run([loss, accuracy],
                                                feed_dict=feed_dict_batch)

                print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".
                    format(iteration, loss_batch, acc_batch))

        # Run validation after every epoch
        feed_dict_valid = {x: x_valid[:1000], y: y_valid[:1000]}
        loss_valid, acc_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
        print('---------------------------------------------------------')
        print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
            format(epoch + 1, loss_valid, acc_valid))
        print('---------------------------------------------------------')