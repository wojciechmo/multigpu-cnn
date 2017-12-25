# MultiGPU CNN

## Usage
Prepare training and testing files - each row should contain class and image path.
```
git clone ‘https://github.com/moxiegushi/pokeGAN.git’
cd multigpu_cnn
python make_tfrecord.py --list-path=./train.txt --tfrecord-path=./train.tfrecord
python make_tfrecord.py --list-path=./test.txt --tfrecord-path=./test.tfrecord
python train.py
```

<img src="https://s14.postimg.org/kp2yzudoh/loss_accuracy.png" width="500">
