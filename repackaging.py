
# coding: utf-8

# In[1]:


import tensorflow as tf
import math

import os

import errno

import shutil



import numpy as np

import matplotlib.pyplot as plt

import os

#import utils


# In[2]:


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# In[3]:



def get_grid_dim(x):

    """

    Transforms x into product of two integers

    :param x: int

    :return: two ints

    """

    factors = prime_powers(x)

    if len(factors) % 2 == 0:

        i = int(len(factors) / 2)

        return factors[i], factors[i - 1]



    i = len(factors) // 2

    return factors[i], factors[i]





def prime_powers(n):

    """

    Compute the factors of a positive integer

    Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python

    :param n: int

    :return: set

    """

    factors = set()

    for x in range(1, int(math.sqrt(n)) + 1):

        if n % x == 0:

            factors.add(int(x))

            factors.add(int(n // x))

    return sorted(factors)





def empty_dir(path):

    """

    Delete all files and folders in a directory

    :param path: string, path to directory

    :return: nothing

    """

    for the_file in os.listdir(path):

        file_path = os.path.join(path, the_file)

        try:

            if os.path.isfile(file_path):

                os.unlink(file_path)

            elif os.path.isdir(file_path):

                shutil.rmtree(file_path)

        except Exception as e:

            print ('Warning: {}'.format(e))





def create_dir(path):

    """

    Creates a directory

    :param path: string

    :return: nothing

    """

    try:

        os.makedirs(path)

    except OSError as exc:

        if exc.errno != errno.EEXIST:

            raise





def prepare_dir(path, empty=False):

    """

    Creates a directory if it soes not exist

    :param path: string, path to desired directory

    :param empty: boolean, delete all directory content if it exists

    :return: nothing

    """

    if not os.path.exists(path):

        create_dir(path)



    if empty:

        empty_dir(path)


# In[4]:


PLOT_DIR = './out/latest_plots'
def plot_conv_weights(weights, name, channels_all=True):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder

    plot_dir = os.path.join(PLOT_DIR, 'conv_weights')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    prepare_dir(plot_dir, empty=True)
    #if not os.path.exists(plot_dir):
    #    os.makedirs(plot_dir)
    w_min = np.min(weights)
    w_max = np.max(weights)

    channels = [0]
    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])

    # get number of convolutional filters
    num_filters = weights.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))
    
    # iterate channels
    for channel in channels:
        # iterate filters inside every channel
        for l, ax in enumerate(axes.flat):
            # get a single filter
            img = weights[:, :, channel, l]
            # put it on the grid
            im = ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='Greys')
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
        if channel == 0:
            fig.colorbar(im, ax=axes.ravel().tolist())   
        
        # save figure
        plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel)), bbox_inches='tight')


# In[5]:


def parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    feature_myTFrecord = {'train/image': tf.FixedLenFeature([], tf.string), #you wrote the files even in the validate set as train BAKAYARO
               'train/label': tf.FixedLenFeature([], tf.int64)}
    
    features = tf.parse_single_example(
        serialized_example,features= feature_myTFrecord)
    
    
    image = tf.decode_raw(features['train/image'], tf.float32)

    # Cast label data into int32
    temp_label = tf.cast(features['train/label'], tf.int32)
    # Reshape image data into the original shape
    image = tf.reshape(image, [575, 400, 1])

    label = tf.cast(tf.subtract(temp_label, 1), tf.int32)

    return image, label


# In[21]:


def plot_conv_output(conv_img, name):

    """

    Makes plots of results of performing convolution

    :param conv_img: numpy array of rank 4

    :param name: string, name of convolutional layer

    :return: nothing, plots are saved on the disk

    """

    # make path to output folder

    plot_dir = os.path.join(PLOT_DIR, 'conv_output')

    plot_dir = os.path.join(plot_dir, name)



    # create directory if does not exist, otherwise empty it

    prepare_dir(plot_dir, empty=True)



    w_min = np.min(conv_img)

    w_max = np.max(conv_img)



    # get number of convolutional filters

    num_filters = conv_img.shape[3]



    # get number of grid rows and columns

    grid_r, grid_c = get_grid_dim(num_filters)



    # create figure and axes

    fig, axes = plt.subplots(min([grid_r, grid_c]),

                             max([grid_r, grid_c]))



    # iterate filters
    for x in range(conv_img.shape[0]):
        for l, ax in enumerate(axes.flat):

        # get a single image

            img = conv_img[x, :, :,  l]

        # put it on the grid

            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys')

        # remove any labels from the axes

            ax.set_xticks([])

            ax.set_yticks([])

    # save figure

        plt.savefig(os.path.join(plot_dir, 'image-{}-{}.png'.format(x,name)), bbox_inches='tight')


# In[7]:


reset_graph()
saver = tf.train.import_meta_graph("./../my_hand_model.meta")
check_conv = tf.get_default_graph().get_tensor_by_name('conv1/Relu:0')
myX = tf.get_default_graph().get_tensor_by_name("inputs/X:0")
myY = tf.get_default_graph().get_tensor_by_name("inputs/y:0")
myAccuracy = tf.get_default_graph().get_tensor_by_name("eval/Mean:0")
myConv1 = tf.get_default_graph().get_tensor_by_name('conv1/Conv2D:0')
weights = tf.get_default_graph().get_tensor_by_name("conv1/kernel:0")
weights_conv2 = tf.get_default_graph().get_tensor_by_name("conv2/kernel:0")
myConv2 = tf.get_default_graph().get_tensor_by_name('conv2/Conv2D:0')


# In[8]:


data_path = 'C:/Users/Gebruiker/PycharmProjects/TFRecords/validate.tfrecords'
#data_path = './../extra.tfrecords'
filenames = data_path
batch_size =12
NUM_PER_EPOCH = 30
# Repeat infinitely.
dataset = tf.data.TFRecordDataset(filenames,buffer_size=2 * batch_size,num_parallel_reads=4)#.repeat()

# Parse records.
dataset = dataset.map(parser)

# Potentially shuffle records.
min_queue_examples = int(NUM_PER_EPOCH * 0.4)
dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)
# Batch it up.
dataset = dataset.batch(batch_size)
    #iterator = dataset.make_one_shot_iterator()
    #image_batch, label_batch = iterator.get_next()
iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
next_element = iterator.get_next()    

training_init_op = iterator.make_initializer(dataset)


# In[9]:


init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


# In[10]:


#n_epochs = 2
allAccs = []
#n_epochs = 5
with tf.Session() as sess:
    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess,coord=coord)
    sess.run(training_init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess,coord=coord)
    saver = tf.train.import_meta_graph("./../my_hand_model.meta")
    saver.restore(sess,tf.train.latest_checkpoint('./../'))
    try:
        X_batch, y_batch = sess.run(next_element)
    except tf.errors.OutOfRangeError:
        print('error has occured!!! on ' , iteration, ' of epoch ',epoch)
        sess.run(training_init_op)
        X_batch, y_batch = sess.run(next_element)
    #print(y_batch)
    #sess.run([training_op], feed_dict={X: X_batch, y: y_batch})
    acc_train = myAccuracy.eval(feed_dict={myX: X_batch, myY: y_batch})
    draw_conv1_conv2D = myConv1.eval(feed_dict={myX: X_batch, myY: y_batch})
    #draw_conv1_relu = check_conv.eval(feed_dict={myX: X_batch, myY: y_batch})
    draw_conv2_conv2D = myConv2.eval(feed_dict={myX: X_batch, myY: y_batch})
    got_weights = sess.run(weights)
    got_weights2 = sess.run(weights_conv2)
    #myCC = myCorrect.eval(feed_dict={myX: X_batch, myY: y_batch})
    print("validation accuracy now:", acc_train)
    #print(epoch, "validation accuracy now:", myCC)
    allAccs.append(acc_train)
    
    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)


# In[11]:


plot_conv_weights(got_weights, 'conv1-weigths-{}'.format(2))

plot_conv_weights(got_weights2, 'conv2-weights{}'.format(2))
# In[12]:


plot_conv_output(draw_conv2_conv2D, 'conv2-images{}'.format(7))


# In[13]:


plot_conv_output(draw_conv1_conv2D, 'conv1-images{}'.format(7))


