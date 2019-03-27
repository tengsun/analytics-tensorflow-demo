import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# define input and output data
INPUT_DATA = './data/flower_photos'
OUTPUT_FILE = './data/flower_processed.npy'

# define test and validation percent
VALIDATION_PERCENT = 10
TEST_PERCENT = 10

# load data and split into training, testing and validation
def create_image_lists(sess, test_percent, validation_percent):
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True

    # init each dataset
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0

    # load all sub dirs
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
    
        # retrieve all images
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        print('current dir:', sub_dir)
        for ext in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + ext)
            file_list.extend(glob.glob(file_glob))
            if not file_list: continue

            # process image file
            for file_name in file_list:
                # read and convert into 299x299
                image_raw_data = gfile.FastGFile(file_name, 'rb').read()
                image = tf.image.decode_jpeg(image_raw_data)
                if image.dtype != tf.float32:
                    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
                image = tf.image.resize_images(image, [299, 299])
                image_value = sess.run(image)
                
                # split dataset randomly
                chance = np.random.randint(100)
                if chance < validation_percent:
                    validation_images.append(image_value)
                    validation_labels.append(current_label)
                elif chance < (test_percent + validation_percent):
                    testing_images.append(image_value)
                    testing_labels.append(current_label)
                else:
                    training_images.append(image_value)
                    training_labels.append(current_label)
            current_label += 1
    
    # shuffle the dataset
    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)

    return np.asarray([training_images, training_labels, validation_images, validation_labels, 
        testing_images, testing_labels])

def main():
    with tf.Session() as sess:
        processed_data = create_image_lists(sess, TEST_PERCENT, VALIDATION_PERCENT)
        np.save(OUTPUT_FILE, processed_data)

if __name__ == '__main__':
    main()
