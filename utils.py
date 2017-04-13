from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from datetime import datetime
import sys
import math
import time
from data import inputs, standardize_image
import numpy as np
import tensorflow as tf

RESIZE_AOI = 256
RESIZE_FINAL = 227

# Read image files            
class ImageCoder(object):
    
    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()
        
        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)
        
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
        self.crop = tf.image.resize_images(self._decode_jpeg, (RESIZE_AOI, RESIZE_AOI))

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})
        
    def decode_jpeg(self, image_data):
        image = self._sess.run(self.crop, #self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})

        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image
        

def _is_png(filename):
    """Determine if a file contains a PNG format image.
    Args:
    filename: string, path of the image file.
    Returns:
    boolean indicating if the image is a PNG.
    """
    return '.png' in filename
        
def make_batch(filename, coder, multicrop):
    """Process a single image file.
    Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'r') as f:
        image_data = f.read()

    # Convert any PNG to JPEG's for consistency.
    if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)
    
    image = coder.decode_jpeg(image_data)

    crops = []

    if multicrop is False:
        print('Running a single image')
        crop = tf.image.resize_images(image, (RESIZE_FINAL, RESIZE_FINAL))
        image = standardize_image(crop)

        crops.append(image)
    else:
        print('Running multi-cropped image')
        h = image.shape[0]
        w = image.shape[1]
        hl = h - RESIZE_FINAL
        wl = w - RESIZE_FINAL

        crop = tf.image.resize_images(image, (RESIZE_FINAL, RESIZE_FINAL))
        crops.append(standardize_image(crop))
        crops.append(tf.image.flip_left_right(crop))

        corners = [ (0, 0), (0, wl), (hl, 0), (hl, wl), (int(hl/2), int(wl/2))]
        for corner in corners:
            ch, cw = corner
            cropped = tf.image.crop_to_bounding_box(image, ch, cw, RESIZE_FINAL, RESIZE_FINAL)
            crops.append(standardize_image(cropped))
            flipped = tf.image.flip_left_right(cropped)
            crops.append(standardize_image(flipped))

    image_batch = tf.stack(crops)
    return image_batch
