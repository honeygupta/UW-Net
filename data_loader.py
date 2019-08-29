import tensorflow as tf

import model
import uwnet_datasets

def _load_samples(csv_name, image_type):
    filename_queue = tf.train.string_input_producer(
        [csv_name])

    reader = tf.TextLineReader()
    _, csv_filename = reader.read(filename_queue)

    record_defaults = [tf.constant([], dtype=tf.string),
                       tf.constant([], dtype=tf.string)]

    filename_i, filename_j = tf.decode_csv(
        csv_filename, record_defaults=record_defaults)

    file_contents_i = tf.read_file(filename_i)
    file_contents_j = tf.read_file(filename_j)
    
    def read_npy_file(item):
        data = np.load(item.decode())
        data1 = np.reshape(data,[480,640,4])
        return data1.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(file_contents_i)
    dataset = dataset.map(lambda item: tuple(tf.py_func(read_npy_file, [item], [tf.float32,])))

    iter = dataset.make_initializable_iterator()
    el = iter.get_next()

    if image_type == '.jpg':
        image_decoded_B = tf.image.decode_jpeg(
            file_contents_j, channels=model.IMG_CHANNELS)
    elif image_type == '.png':
        image_decoded_B = tf.image.decode_png(
            file_contents_j, channels=model.IMG_CHANNELS, dtype=tf.uint8)

    return el, image_decoded_B
