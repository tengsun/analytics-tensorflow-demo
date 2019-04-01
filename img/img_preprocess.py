import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile('./data/swk_raw.jpg', 'rb').read()

def decode_encode_img():
    with tf.Session() as sess:
        # decode image
        img_data = tf.image.decode_jpeg(image_raw_data)
        print(img_data.eval())
    
        # show on plot
        plt.imshow(img_data.eval())
        plt.show()

        # encode image
        encode_write_img(img_data, './data/swk_encoded.jpg')

def adjust_img_size():
    with tf.Session() as sess:
        img_data = tf.image.decode_jpeg(image_raw_data)

        # convert to float
        img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
        print(img_data.eval())

        # resize image
        resized_img = tf.image.resize_images(img_data, [50,50], method=0)
        resized_img = tf.image.convert_image_dtype(resized_img, dtype=tf.uint8)
        encode_write_img(resized_img, './data/swk_resized.jpg')

        # crop image
        cropped_img = tf.image.crop_to_bounding_box(img_data, 10, 30, 50, 50)
        cropped_img = tf.image.convert_image_dtype(cropped_img, dtype=tf.uint8)
        encode_write_img(cropped_img, './data/swk_cropped.jpg')

def flip_transpose_img():
    with tf.Session() as sess:
        img_data = tf.image.decode_jpeg(image_raw_data)

        # flip image
        flipped_img = tf.image.flip_left_right(img_data)
        encode_write_img(flipped_img, './data/swk_flipped.jpg')

        # transpose image
        transposed_img = tf.image.transpose_image(img_data)
        encode_write_img(transposed_img, './data/swk_transposed.jpg')

def adjust_img_bchs():
    with tf.Session() as sess:
        img_data = tf.image.decode_jpeg(image_raw_data)

        # adjust different aspects
        adjusted = tf.image.adjust_brightness(img_data, 0.5)
        encode_write_img(adjusted, './data/swk_adjusted_1.jpg')
        adjusted = tf.image.adjust_contrast(img_data, 1)
        encode_write_img(adjusted, './data/swk_adjusted_2.jpg')
        adjusted = tf.image.adjust_hue(img_data, 1)
        encode_write_img(adjusted, './data/swk_adjusted_3.jpg')
        adjusted = tf.image.adjust_saturation(img_data, 5)
        encode_write_img(adjusted, './data/swk_adjusted_4.jpg')
        img_data = tf.image.convert_image_dtype(img_data, dtype=tf.uint8)
        adjusted = tf.image.per_image_standardization(img_data)
        adjusted = tf.image.convert_image_dtype(adjusted, dtype=tf.uint8)
        encode_write_img(adjusted, './data/swk_adjusted_5.jpg')

def encode_write_img(img, path):
    encoded_image = tf.image.encode_jpeg(img)
    with tf.gfile.GFile(path, 'wb') as f:
        f.write(encoded_image.eval())


if __name__ == '__main__':
    # decode_encode_img()
    # adjust_img_size()
    # flip_transpose_img()
    adjust_img_bchs()
