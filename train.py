import tensorflow as tf
import numpy as np
import os

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH = 128, 128, 3
CONV1_DEPTH, CONV1_SIZE, CONV1_STRIDE, CONV1_DROPOUT, POOL1_SIZE, POOL1_STRIDE = 16, 9, 1, 0.5, 2, 2
CONV2_DEPTH, CONV2_SIZE, CONV2_STRIDE, CONV2_DROPOUT, POOL2_SIZE, POOL2_STRIDE = 32, 7, 1, 0.6, 2, 2
CONV3_DEPTH, CONV3_SIZE, CONV3_STRIDE, CONV3_DROPOUT, POOL3_SIZE, POOL3_STRIDE = 64, 5, 1, 0.7, 2, 2
FC1_NUM, FC1_DROPOUT = 64, 0.8
FC2_NUM = 10

NUM_GPUS = 2
BATCH_SIZE = 100
TRAINING_ITER = 100000
WEIGHT_DECAY = 0.0001 
STARTER_LR, DECAY_RATE, DECAY_STEPS = 0.01, 0.9, 1000
SAVE_INTERVAL, TEST_INTERVAL, TEST_BATCHES = 10000, 2000, 10
TRAIN_TFRECORD, TEST_TFRECORD  = './train.tfrecord', './test.tfrecord'
SAVE_GRAPH, GRAPH_PATH, MODEL_PATH = True, './graph', './model'

# ---------------------------------------------------------------------------------
# -------------------------------- tfrecord reader --------------------------------
# ---------------------------------------------------------------------------------

def read_example(filename, batch_size):

	reader = tf.TFRecordReader()
	filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
	_, serialized_example = reader.read(filename_queue)

	parsed_example = tf.parse_single_example(serialized_example,features={'image': tf.FixedLenFeature([], tf.string),
												'label': tf.FixedLenFeature([], tf.int64)})

	image_raw = tf.decode_raw(parsed_example['image'], tf.uint8)
	image = tf.cast(tf.reshape(image_raw, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH]), tf.float32)
	label_raw = parsed_example['label']
	label = tf.one_hot(tf.cast(label_raw, tf.int32), depth=FC2_NUM)

	return image,label

def preprocess(image, train_logical=False):

	image=image/255.0

	if train_logical == True:

		image = tf.image.random_brightness(image, max_delta=0.1)
		image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
		image = tf.image.random_flip_left_right(image)

	return image

def make_batch(image, label, batch_size):

	min_queue_examples = 500

	batch_images, batch_labels = tf.train.shuffle_batch([image,label], batch_size=batch_size, 
														capacity=min_queue_examples+100*batch_size,
														min_after_dequeue=min_queue_examples, num_threads=8)

	return batch_images, batch_labels

def read_record(filename, batch_size, train_logical=False):
	
	image,label = read_example(filename,batch_size)
	image = preprocess(image,train_logical)
	batch_images, batch_labels = make_batch(image,label,batch_size)

	return batch_images, batch_labels

# ---------------------------------------------------------------------------------
# --------------------------------- net structure ---------------------------------
# ---------------------------------------------------------------------------------

def variable_weight_decay(name, shape, initializer, regularizer):

	with tf.device('/cpu:0'):

		var = tf.get_variable(name, shape, initializer=initializer, regularizer=regularizer, dtype=tf.float32)

	return var

def conv_layer(input, conv_size, in_depth, out_depth, conv_stride, pool_size, pool_stride, dropout, w_initializer, b_initializer, regularizer, train_logical, name):

	with tf.variable_scope(name) as scope:
		
		w = variable_weight_decay('w', [conv_size, conv_size, in_depth, out_depth], w_initializer, regularizer)
		b = variable_weight_decay('b', [out_depth] , b_initializer, regularizer)

		conv = tf.nn.conv2d(input, w, strides=[1, conv_stride, conv_stride,1], padding='SAME',name='conv')
		conv = tf.add(conv, b, name='add')
		relu = tf.nn.relu(conv, name='relu')
		pool = tf.nn.max_pool(relu, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_stride, pool_stride, 1], padding='SAME', name='max_pool')
	
		if train_logical == True:
			keep_prob = tf.constant(dropout, name='keep_prob')
			pool = tf.nn.dropout(pool, keep_prob, name='dropout')

	return pool

def fc_layer(input, in_num, out_num,dropout, w_initializer, b_initializer, regularizer, train_logical, name, reshape, not_final):

	with tf.variable_scope(name) as scope:
		
		w = variable_weight_decay('w', [in_num, out_num], w_initializer, regularizer)
		b = variable_weight_decay('b', [out_num] , b_initializer, regularizer)
		
		if reshape == True:
			input = tf.reshape(input, [-1, in_num])
	
		fc = tf.matmul(input, w, name='matmul')
		fc = tf.add(fc, b, name='add')
	
		if not_final == True:

			fc = tf.nn.relu(fc,name='relu')

			if train_logical == True:
				keep_prob = tf.constant(dropout, name='keep_prob')
				fc = tf.nn.dropout(fc, keep_prob, name='dropout')

	return fc	

def tower_inference(images, train_logical=False):

	xavier_initializer_conv = tf.contrib.layers.xavier_initializer_conv2d()
	xavier_initializer_fc = tf.contrib.layers.xavier_initializer()
	zeros_initializer = tf.zeros_initializer()
	
	regularizer = tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY)

	pool1 = conv_layer(images, CONV1_SIZE, IMAGE_DEPTH, CONV1_DEPTH, CONV1_STRIDE, POOL1_SIZE, POOL1_STRIDE, CONV1_DROPOUT,
				        xavier_initializer_conv, zeros_initializer, regularizer, train_logical, 'conv1')

	pool2 = conv_layer(pool1, CONV2_SIZE, CONV1_DEPTH, CONV2_DEPTH, CONV2_STRIDE, POOL2_SIZE, POOL2_STRIDE, CONV2_DROPOUT,
				        xavier_initializer_conv, zeros_initializer, regularizer, train_logical, 'conv2')

	pool3 = conv_layer(pool2, CONV3_SIZE, CONV2_DEPTH, CONV3_DEPTH, CONV3_STRIDE, POOL3_SIZE, POOL3_STRIDE, CONV3_DROPOUT,
				        xavier_initializer_conv, zeros_initializer, regularizer, train_logical, 'conv3')

	pool3_num = int(np.prod(pool3.get_shape()[1:]))

	fc1 = fc_layer(pool3, pool3_num, FC1_NUM, FC1_DROPOUT, xavier_initializer_fc,zeros_initializer, regularizer, 
				    train_logical, 'fc1', reshape=True, not_final=True)

	y = fc_layer(fc1, FC1_NUM, FC2_NUM, None, xavier_initializer_fc,zeros_initializer, regularizer, 
				    train_logical, 'fc2', reshape=False, not_final=False)

	return y

def tower_loss(logits, labels):

	cross_entropy_single = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
	cross_entropy = tf.reduce_mean(cross_entropy_single, name='cross_entropy')
	
	reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	reg_term = tf.add_n(reg_variables, name='reg_term')
	
	return tf.add_n([cross_entropy, reg_term], name='loss')
	
def tower_accuracy(logits,labels):
	
	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1), name='correct_prediction')
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
	
	return accuracy
	
def get_variables():
	
	variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	
	variables_w = [var for var in variables if 'w:0' in var.name]
	variables_b = [var for var in variables if 'b:0' in var.name]
	
	return variables_w, variables_b
	
def tower_grads(loss):
	
	variables_w, variables_b = get_variables()
	
	grads_w = tf.gradients(loss, variables_w,name = 'grads_w')
	grads_b = tf.gradients(loss, variables_b,name = 'grads_b')
	
	return grads_w, grads_b

def merge_grads(tower_grads):

	grads_transposed = [list(i) for i in zip(*tower_grads)]

	grads_merged = []

	for gpu_grads in grads_transposed:
		grads_merged.append(tf.add_n(gpu_grads))

	return grads_merged
	
def merge_accuracies(tower_accuracies):

	accuracies_merged = tf.add_n(tower_accuracies)/float(len(tower_accuracies))
	
	return accuracies_merged
	
def merge_losses(tower_losses):

	losses_merged = tf.add_n(tower_losses)
	
	return losses_merged

# ----------------------------------------------------------------------------
# --------------------------------- training ---------------------------------
# ----------------------------------------------------------------------------

def make_dir(directory):

	if not os.path.exists(directory):
		os.makedirs(directory)

def train():

	with tf.device('/cpu:0'):

		with tf.name_scope('train_batch'):
			train_images,train_labels = read_record(TRAIN_TFRECORD, BATCH_SIZE, train_logical=True)
			train_batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([train_images, train_labels], capacity=2*NUM_GPUS)

		with tf.name_scope('test_batch'):	
			test_images,test_labels = read_record(TEST_TFRECORD, BATCH_SIZE)
			test_batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([test_images, test_labels], capacity=2*NUM_GPUS)

		net_grads_w, net_grads_b, net_losses, net_accuracies, net_summaries = [], [], [], [], []

		with tf.variable_scope('net') as scope:
			for i in range(NUM_GPUS):  
				with tf.device('/gpu:%d' % i):
					with tf.name_scope('tower_%d' % i) as name_scope:

						train_batch_images, train_batch_labels = train_batch_queue.dequeue()
						test_batch_images, test_batch_labels = test_batch_queue.dequeue()
			
						with tf.device('/cpu:0'):
							net_summaries.append(tf.summary.image(name_scope+'train_images', train_batch_images, max_outputs=10))

						logits = tower_inference(train_batch_images, train_logical=True)
						loss = tower_loss(logits, train_batch_labels)
						net_losses.append(loss)
						
						tf.get_variable_scope().reuse_variables()
						
						test_logits = tower_inference(test_batch_images, train_logical=False)
						accuracy = tower_accuracy(test_logits, test_batch_labels)
						net_accuracies.append(accuracy)
						
						grads_w,grads_b = tower_grads(loss)
						net_grads_w.append(grads_w) 
						net_grads_b.append(grads_b)	
						
						tf.get_variable_scope().reuse_variables()

		with tf.name_scope('optimizer'):

			global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
			learning_rate = tf.train.exponential_decay(STARTER_LR, global_step, DECAY_STEPS, DECAY_RATE, staircase=True)

			grads_w_merged = merge_grads(net_grads_w)
			grads_b_merged = merge_grads(net_grads_b)

			variables_w, variables_b = get_variables()
			
			opt_w = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam_w')
			opt_b = tf.train.AdamOptimizer(learning_rate=learning_rate*2, name='Adam_b')
								
			train_op_w = opt_w.apply_gradients(zip(grads_w_merged, variables_w))
			train_op_b = opt_b.apply_gradients(zip(grads_b_merged, variables_b))

			increment = tf.assign_add(global_step, 1, name='increment')
			train_step = tf.group(train_op_w, train_op_b, increment)

		with tf.name_scope('loss'):

			total_loss = merge_losses(net_losses)
		
		with tf.name_scope('accuracy'):

			test_step = merge_accuracies(net_accuracies)

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
	config = tf.ConfigProto(gpu_options=gpu_options)
	sess = tf.Session(config=config)

	if SAVE_GRAPH == True:
		make_dir(GRAPH_PATH)
		summary_writer = tf.summary.FileWriter(GRAPH_PATH, sess.graph)

	net_summaries.append(tf.summary.scalar('learning_rate', learning_rate))
	net_summaries.append(tf.summary.scalar('loss', total_loss))

	for grad_w, var_w in zip(grads_w_merged,variables_w):
		net_summaries.append(tf.summary.histogram(var_w.name + '/gradients', grad_w))

	for grad_b, var_b in zip(grads_b_merged,variables_b):
		net_summaries.append(tf.summary.histogram(var_b.name + '/gradients', grad_b))

	for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
		net_summaries.append(tf.summary.histogram(var.name, var))

	test_accuracy = tf.placeholder(tf.float32, name='accuracy_placeholder')
	net_summaries.append(tf.summary.scalar('test_accuracy', test_accuracy))

	summary_op = tf.summary.merge(net_summaries)

	sess.run(tf.global_variables_initializer())
	coord = tf.train.Coordinator()
	saver = tf.train.Saver()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	for i in range(TRAINING_ITER):
		
		sess.run(train_step)

		if (i+1)%TEST_INTERVAL == 0:
			
			data_acc = 0	
			for j in range(TEST_BATCHES):
				data_acc_batch = sess.run(test_step)
				data_acc += data_acc_batch
			data_acc /= float(TEST_BATCHES)
			
			if SAVE_GRAPH == True:

				summary_str = sess.run(summary_op, feed_dict={test_accuracy: data_acc})
				summary_writer.add_summary(summary_str, i)

			print 'accuracy:', data_acc
			
		if (i+1)%SAVE_INTERVAL == 0:

			make_dir(MODEL_PATH)
			saver.save(sess, os.path.join(MODEL_PATH, 'net'), global_step=i+1)

	saver.save(sess, os.path.join(MODEL_PATH,'net'), global_step=i+1)

	sess.close()
	coord.request_stop()
	coord.join(threads)

if __name__ == "__main__":
    train()
