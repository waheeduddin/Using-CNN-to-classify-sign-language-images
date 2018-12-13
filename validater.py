
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


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


# In[3]:


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# In[4]:


reset_graph()


# In[5]:


#data_path = './extra.tfrecords'
data_path = 'C:/Users/Gebruiker/PycharmProjects/TFRecords/validate.tfrecords'
filenames = data_path
batch_size =15
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


# In[6]:


saver = tf.train.import_meta_graph("my_hand_model.meta")


# In[7]:


myX = tf.get_default_graph().get_tensor_by_name("inputs/X:0")
myY = tf.get_default_graph().get_tensor_by_name("inputs/y:0")
myAccuracy = tf.get_default_graph().get_tensor_by_name("eval/Mean:0")


# In[8]:


init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


# In[ ]:


n_epochs = 3
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
    saver = tf.train.import_meta_graph("my_hand_model.meta")
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    for epoch in range(n_epochs):
        for iteration in range(50):
            print('At epoch: ', epoch,' iteartion: ',iteration)
            try:
                X_batch, y_batch = sess.run(next_element)
            except tf.errors.OutOfRangeError:
                print('error has occured!!! on ' , iteration, ' of epoch ',epoch)
                sess.run(training_init_op)
                X_batch, y_batch = sess.run(next_element)
            #print(y_batch)
            #sess.run([training_op], feed_dict={X: X_batch, y: y_batch})
            acc_train = myAccuracy.eval(feed_dict={myX: X_batch, myY: y_batch})
            #myCC = myCorrect.eval(feed_dict={myX: X_batch, myY: y_batch})
            print(epoch, "validation accuracy now:", acc_train)
            #print(epoch, "validation accuracy now:", myCC)
            allAccs.append(acc_train)
    
    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)


# In[11]:


with open('answer.txt','w') as thefile:
    for item in allAccs:
        thefile.write("%s\n" % item)
    thefile.write("%s\n" % np.mean(allAccs))

