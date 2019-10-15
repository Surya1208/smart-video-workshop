export CHECKPOINT_DIR=/tmp/$USER/checkpoints

mkdir $CHECKPOINT_DIR

wget http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz

tar -xvf inception_v1_2016_08_28.tar.gz

mv inception_v1.ckpt $CHECKPOINT_DIR

rm inception_v1_2016_08_28.tar.gz
