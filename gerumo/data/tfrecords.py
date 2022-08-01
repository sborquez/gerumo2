import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import tensorflow as tf

from ..config.config import CfgNode as CN
from ..utils.structures import InputShape, Task
from .generators import GENERATOR_REGISTRY


def observation_to_example(xi, yi, input_shape):
    observation = {}
    if input_shape.has_image() and input_shape.has_features():
        x_img, x_feat = xi
        observation['image'] = tf.train.Feature(float_list=tf.train.FloatList(value=x_img[0].flatten().tolist()))
        observation['features'] = tf.train.Feature(float_list=tf.train.FloatList(value=x_feat[0].flatten().tolist()))
    elif input_shape.has_image():
        x_img = xi
        observation['image'] = tf.train.Feature(float_list=tf.train.FloatList(value=x_img[0].flatten().tolist()))
    elif input_shape.has_features():
        x_feat = xi
        observation['features'] = tf.train.Feature(float_list=tf.train.FloatList(value=x_feat[0].flatten().tolist()))
    observation['y'] = tf.train.Feature(float_list=tf.train.FloatList(value=yi[0].tolist()))
    return tf.train.Example(features=tf.train.Features(feature=observation))


def generator_to_record(dataset_generator, dataset_split, output_dir, samples_per_file, input_shape):
    assert input_shape.batch_size == 1
    for i, (x, y) in enumerate(dataset_generator):
        if i % samples_per_file == 0:
            if i != 0:
                writer.close()
            elif input_shape is not None:
                shape_file = os.path.join(output_dir, f'input_shape.json')
                with open(shape_file, 'w') as f:
                    input_shape.dump(f)
            record_file = os.path.join(output_dir, f'{dataset_split}_{i//samples_per_file:04}.tfrecords')
            writer = tf.io.TFRecordWriter(record_file)
        tf_example = observation_to_example(x, y, input_shape)
        writer.write(tf_example.SerializeToString())


def _task(generator, start, end, record_file, input_shape):
    writer = tf.io.TFRecordWriter(record_file)
    for i in range(start, end):
        x, y = generator[i]
        tf_example = observation_to_example(x, y, input_shape)
        writer.write(tf_example.SerializeToString())
    writer.close()


def generator_to_record_mp(dataset_generator, dataset_split, output_dir, samples_per_file, input_shape, workers=25):
    assert input_shape.batch_size == 1
    shape_file = os.path.join(output_dir, f'input_shape.json')
    with open(shape_file, 'w') as f:
        input_shape.dump(f)
    args = []
    for i in range(0, len(dataset_generator), samples_per_file):
        args.append((
            dataset_generator,
            i,
            min(i+samples_per_file, len(dataset_generator)),
            os.path.join(output_dir, f'{dataset_split}_{i//samples_per_file:04}.tfrecords'),
            input_shape
        ))
    with Pool(workers) as pool:
        pool.starmap(_task, args)


def example_to_observation(example, input_shape, num_targets):
    tfrecord_format = {}
    if input_shape.has_image():
        tfrecord_format['image'] = tf.io.FixedLenFeature([np.prod(input_shape.images_shape)], tf.float32)
    if input_shape.has_features():
        tfrecord_format['features'] = tf.io.FixedLenFeature([np.prod(input_shape.features_shape)], tf.float32)
    tfrecord_format['y'] = tf.io.FixedLenFeature([num_targets], tf.float32)
    tfrecord_format = (tfrecord_format)    
    example = tf.io.parse_single_example(example, tfrecord_format)
    
    if input_shape.has_image() and input_shape.has_features():
        x_img = tf.reshape(example['image'], input_shape.images_shape[1:])
        x_feat = tf.reshape(example['features'], input_shape.features_shape[1:])
        xi = (x_img, x_feat)
    elif input_shape.has_image():
        xi = tf.reshape(example['image'], input_shape.images_shape[1:])
    elif input_shape.has_features():
        xi = tf.reshape(example['features'], input_shape.features_shape[1:])
    yi = example['y']
    return xi, yi


@GENERATOR_REGISTRY.register()
def build_tf_record_dataset(cfg: CN, dataset: pd.DataFrame, subset: str) -> tf.data.Dataset:
    """Use precomputed dataset in the tfRecord format.

    Args:
        cfg (CfgNode): Loaded datasets. It must contain the "tfRecords_folder"
        key-values on the GENERATOR.KWARGS pointing to the tfRecords' folder.
        
        dataset (pd.DataFrame): Tabular reference dataset.
        subset (str): Name of the subset, the prefix of the tfRecords files.

    Raises:
        ValueError: The configuraton doesn't contain "tfRecords_folder".

    Returns:
        tf.data.Dataset: The loaded dataset.
    """
    # Parse generator keyword arguments
    generator_kwargs = { k:v for k,v in cfg.GENERATOR.KWARGS }
    ## Find data folder
    tfRecord_folder = generator_kwargs.get('tfRecords_folder', None)
    if tfRecord_folder is None:
        raise ValueError('"tfRecords_folder" key not in "cfg.GENERATOR.KWARGS"')
    pattern = os.path.join(tfRecord_folder, f'{subset}_[0-9]*.tfrecords')
    tf_record_files = tf.io.gfile.glob(pattern)
    ## Find input shape file
    input_shape_file = os.path.join(tfRecord_folder, 'input_shape.json')
    if not os.path.exists(input_shape_file):
        raise ValueError('"input_shape_file" doesn\'t exist in "tfRecords_folder".')
    # Setup generator
    ## Input tensors shapes
    autotune = tf.data.AUTOTUNE
    batch_size = cfg.SOLVER.BATCH_SIZE
    dataset_size = int(np.floor(len(dataset) / batch_size))
    with open(input_shape_file) as f:
        input_shape = InputShape.load(f)
    ## Output tensor shape
    if Task[cfg.MODEL.TASK] is Task.REGRESSION:
        num_targets = len(cfg.OUTPUT.REGRESSION.TARGETS) 
    else:
        raise NotImplementedError(Task[cfg.MODEL.TASK])
    # Instance the dataset
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = cfg.DETERMINISTIC # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        tf_record_files
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(example_to_observation, input_shape=input_shape, num_targets=num_targets),
        num_parallel_calls=autotune
    )
    dataset = dataset.shuffle(2048)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=autotune)
    dataset = dataset.batch(batch_size, drop_remainder=True) 
    input_shape.set_batch_size(batch_size)
    return dataset, input_shape, dataset_size
    