# Multiple GPU Convolutional Neural Network

Clear multiple GPU Convolutional Neural Network trainig imlementation with Tensorflow.

<img src="https://github.com/WojciechMormul/multigpu-cnn/blob/master/imgs/full.png" width="250">
<img src="https://github.com/WojciechMormul/multigpu-cnn/blob/master/imgs/train_batch.png" width="300"/> 

<img src="https://github.com/WojciechMormul/multigpu-cnn/blob/master/imgs/net.png" width="600">
<img src="https://github.com/WojciechMormul/multigpu-cnn/blob/master/imgs/tower.png" width="800">
<img src="https://github.com/WojciechMormul/multigpu-cnn/blob/master/imgs/optimizer.png" height="300">

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
