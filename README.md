# Multiple GPU Convolutional Neural Network

Clear multiple GPU Convolutional Neural Network trainig imlementation with Tensorflow.

<img src="https://s14.postimg.org/86qywso0h/full.png" width="250">
<img src="https://s14.postimg.org/3kuuok535/train_batch.png" width="300"/> 

<img src="https://s14.postimg.org/ovsesoab5/net.png" width="600">
<img src="https://s14.postimg.org/jo7bo0jdd/tower.png" width="800">
<img src="https://s14.postimg.org/mjkeuwpy9/optimizer.png" height="300">

## Usage
Prepare text files for training and testing - each row should contain class and image path.
```
git clone https://github.com/WojciechMormul/multigpu_cnn.git
cd multigpu_cnn
python make_tfrecord.py --list-path=./train.txt --tfrecord-path=./train.tfrecord
python make_tfrecord.py --list-path=./test.txt --tfrecord-path=./test.tfrecord
python train.py
```
<img src="https://s14.postimg.org/73297bqwx/sum.png" width="600">
