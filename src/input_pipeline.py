"""ImageNet input pipeline. Codes are adopted from
https://github.com/google/flax/blob/main/examples/imagenet/input_pipeline.py
"""
import tensorflow as tf
import tensorflow_datasets as tfds


IMAGE_SIZE = 224
CROP_PADDING = 32


def _distorted_bounding_box_crop(
        image,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(0.75, 1.33),
        area_range=(0.05, 1.0),
        max_attempts=100
    ): # pylint: disable=too-many-arguments
    """Generates cropped_image using one of the bboxes randomly distorted."""
    shape = tf.shape(image)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box
    image = tf.slice(image, bbox_begin, bbox_size)
    return image


def _at_least_x_are_equal(a, b, x):
    """At least `x` of `a` and `b` `Tensors` are equal."""
    match = tf.equal(a, b)
    match = tf.cast(match, tf.int32)
    return tf.greater_equal(tf.reduce_sum(match), x)


def _resize(image, image_size):
    return tf.image.resize(
        [image], [image_size, image_size],
        method=tf.image.ResizeMethod.BICUBIC)[0]


def _random_flip(image):
    """Random horizontal flip of image."""
    return tf.image.random_flip_left_right(image)


def _random_crop(image, image_size):
    """Make a random crop of image_size."""
    original_shape = tf.shape(image)
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    cropped_image = _distorted_bounding_box_crop(
        image,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(3. / 4, 4. / 3.),
        area_range=(0.08, 1.0),
        max_attempts=10)
    bad = _at_least_x_are_equal(original_shape, tf.shape(cropped_image), 3)

    return tf.cond(
        bad,
        lambda: _center_crop(image, image_size),
        lambda: _resize(cropped_image, image_size))


def _center_crop(image, image_size):
    """Crops to center of image with padding then scales image_size."""
    shape = tf.shape(image)
    image_h = shape[0]
    image_w = shape[1]

    padded_center_crop_size = tf.cast((
        (image_size / (image_size + CROP_PADDING)) *
            tf.cast(tf.minimum(image_h, image_w), tf.float32)), tf.int32)
    offset_h = ((image_h - padded_center_crop_size) + 1) // 2
    offset_w = ((image_w - padded_center_crop_size) + 1) // 2

    bbox_begin = tf.stack([
        offset_h,
        offset_w,
        tf.constant(0, dtype=tf.int32)])
    bbox_size = tf.stack([
        padded_center_crop_size,
        padded_center_crop_size,
        tf.constant(-1, dtype=tf.int32)])

    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return _resize(cropped_image, image_size)


def create_trn_split(
        data_builder,
        batch_size,
        split='train',
        dtype=tf.float32,
        image_size=IMAGE_SIZE,
        cache=True
    ): # pylint: disable=too-many-arguments
    """Returns iterator for training with augmentation."""
    data = data_builder.as_dataset(
        split=split, shuffle_files=True,
        decoders={'image': tfds.decode.SkipDecoding()})
    image_decoder = data_builder.info.features['image'].decode_example
    shuffle_buffer_size = min(
        16*batch_size, data_builder.info.splits[split].num_examples)

    def decode_example(example):
        image = image_decoder(example['image'])
        image = _random_crop(image, image_size)
        image = _random_flip(image)
        image = tf.reshape(image, [image_size, image_size, 3])
        image = tf.cast(image, dtype=dtype)
        return {'images': image, 'labels': example['label']}

    if cache:
        data = data.cache()
    data = data.repeat()
    data = data.shuffle(shuffle_buffer_size)
    data = data.map(decode_example, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.batch(batch_size, drop_remainder=True)
    data = data.prefetch(tf.data.AUTOTUNE)
    return data


def create_val_split(
        data_builder,
        batch_size,
        split='validation',
        dtype=tf.float32,
        image_size=IMAGE_SIZE,
        cache=True
    ): # pylint: disable=too-many-arguments
    """Returns iterator for evaluation without augmentation."""
    data = data_builder.as_dataset(
        split=split, shuffle_files=False,
        decoders={'image': tfds.decode.SkipDecoding()})
    image_decoder = data_builder.info.features['image'].decode_example

    def decode_example(example):
        image = image_decoder(example['image'])
        image = _center_crop(image, image_size)
        image = tf.reshape(image, [image_size, image_size, 3])
        image = tf.cast(image, dtype=dtype)
        return {'images': image, 'labels': example['label']}

    if cache:
        data = data.cache()
    data = data.map(decode_example, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.batch(batch_size, drop_remainder=False)
    data = data.repeat()
    data = data.prefetch(tf.data.AUTOTUNE)
    return data
