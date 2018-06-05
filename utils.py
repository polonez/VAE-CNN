import os
from glob import glob
import tarfile
import tensorflow as tf
import numpy as np
import sklearn.preprocessing as prep
import imageio
from PIL import Image
from tqdm import tqdm


def _float32_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def min_max_scale(d):
    scaler = prep.MinMaxScaler()
    train_shape = d.shape
    d = d.reshape((train_shape[0], -1))
    scaler.fit(d)
    d = scaler.transform(d)
    return d.reshape(train_shape)
# X_train = min_max_scale(np.array(X_train).astype(np.float32))


def write_to_tfrecord(raw_data_pattern='data/*.jpg', tfrecord_name='tfrecord/celebA', batch_size=1024):
    globbed = glob(raw_data_pattern)
    imgs = [imageio.imread(path) for path in tqdm(globbed)]
    batches = [imgs[i:i + batch_size] for i in range(0, len(imgs), batch_size)]

    for shard, batch_img in tqdm(enumerate(batches)):
        output_filename = '{}-{:04d}-of-{:04d}'.format(tfrecord_name, shard, len(batches))
        with tf.python_io.TFRecordWriter(output_filename) as tf_writer:
            for img in batch_img:
                feature = {
                    'raw_image': _bytes_feature(img.tobytes()),
                    'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=img.shape))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                tf_writer.write(example.SerializeToString())


def read_from_jpg(file_pattern="data/*.jpg", batch_size=128, partial=None):
    globbed = glob(file_pattern)
    imgs = []
    if partial:
        imgs = [imageio.imread(path) for path in tqdm(globbed[:partial])]
    else:
        imgs = [imageio.imread(path) for path in tqdm(globbed)]
    imgs = np.array(imgs) / 255.0
    batches = [imgs[i:i + batch_size] for i in range(0, len(imgs), batch_size)]
    return batches, len(imgs)


def read_from_tar_gz(compressed_filename, batch_size=128):
    imgs = []
    with tarfile.open(compressed_filename, 'r:gz') as tfile:
        for mem in tqdm(tfile.getmembers()):
            if mem.name.lower().endswith('jpg'):
                e = tfile.extractfile(mem)
                imgs.append(imageio.imread(e))
    imgs = np.array(imgs) / 255.0
    batches = [imgs[i: i + batch_size] for i in range(0, len(imgs), batch_size)]
    return batches, len(imgs)


def read_from_tfrecord(file_pattern="tfrecord/celebA-*-of-*"):
    read_imgs = []
    cnt = 0
    records = glob(os.path.join("./", file_pattern))
    for record in tqdm(records):
        record_iterator = tf.python_io.tf_record_iterator(path=record)
        s = []
        for record_string in list(record_iterator):
            e = tf.train.Example()
            e.ParseFromString(record_string)
            img = Image.frombytes('RGB', (64, 64), e.features.feature["raw_image"].bytes_list.value[0])
            s.append(np.array(img, dtype=np.float32))
            cnt += 1
        read_imgs.append(min_max_scale(np.array(s)))
    return read_imgs, cnt


if __name__ == '__main__':
    write_to_tfrecord()
