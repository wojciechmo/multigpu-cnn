# Multiple GPU Convolutional Neural Network

Clear multiple GPU Convolutional Neural Network trainig imlementation with Tensorflow.

## Usage
Prepare text files for training and testing - each row should contain class and image path.
```
git clone https://github.com/WojciechMormul/multigpu_cnn.git
cd multigpu_cnn
python make_tfrecord.py --list-path=./train.txt --tfrecord-path=./train.tfrecord
python make_tfrecord.py --list-path=./test.txt --tfrecord-path=./test.tfrecord
python train.py
```
<img src="https://github.com/WojciechMormul/multigpu-cnn/blob/master/imgs/sum.png" width="600">
