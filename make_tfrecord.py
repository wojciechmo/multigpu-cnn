import tensorflow as tf
import argparse
import cv2

p = argparse.ArgumentParser()
p.add_argument("--list-path", required=True, type=str, help='input file with (label,image) pairs')
p.add_argument("--tfrecord-path", required=True, type=str,  help='output TFRecord')
p.add_argument("--depth", required=False, type=int, choices=set((3, 1)), default=3, help='image output depth')
p.add_argument("--resize", required=False, type=str, help='image output size wxh')
args = p.parse_args()

resize = args.resize
depth = args.depth
list_path = args.list_path
tfrecord_path = args.tfrecord_path

if resize is not None:
	width, height=[int(val) for val in resize.split('x')]

with open(list_path, 'r') as list_file:
	with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
		for line in list_file.read().splitlines():
			
			label, path = line.split()
			img = cv2.imread(path)
			
			if depth == 1:
				img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			if resize is not None:
				img = cv2.resize(img, (width, height))

			image_raw = img.tostring()
			example = tf.train.Example(features=tf.train.Features(feature={
				'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
				'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))}))
				
			writer.write(example.SerializeToString())
