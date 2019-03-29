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

def encode_write_img(img, path):
    encoded_image = tf.image.encode_jpeg(img)
    with tf.gfile.GFile(path, 'wb') as f:
        f.write(encoded_image.eval())


if __name__ == '__main__':
    # decode_encode_img()
    adjust_img_size()
