from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six.moves
from datetime import datetime
import sys
import math
import time
from data import inputs, standardize_image
import numpy as np
import tensorflow as tf
#from detect import *
import re
import base64
import cv2

RESIZE_AOI = 256
RESIZE_FINAL = 227

# Modifed from here
# http://stackoverflow.com/questions/3160699/python-progress-bar#3160819
class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='='):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def update(self, step=1):
        self.current += step
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        six.print_('\r' + self.fmt % args, end='')

    def done(self):
        self.current = self.total
        self.update(step=0)
        print('')

# Read image files            
class ImageCoder(object):
    
    def __init__(self):
        # Create a single Session to run all image coding calls.
        config = tf.ConfigProto(allow_soft_placement=True)
        self._sess = tf.Session(config=config)
        
        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)
        
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
        self.crop = tf.image.resize_images(self._decode_jpeg, (RESIZE_AOI, RESIZE_AOI))

        ''' Added to handle memory leak-single look '''
        self.crop_image = tf.image.resize_images(self.crop, (RESIZE_FINAL, RESIZE_FINAL))
        self.image_standradisation = tf.image.per_image_standardization(self.crop_image)
	    self.num_img = None
        self.images_single = tf.placeholder(dtype=tf.float32, shape=(self.num_img, 227, 227, 3))
        self.image_batch_single = tf.stack(self.images_single)
	       
        '''Added to handle memory leak-multi look'''                
        self.standardize_image_holder = tf.placeholder(dtype=tf.float32, shape=(227, 227, 3))
        self.flipped_image_holder = tf.placeholder(dtype=tf.float32, shape=(227, 227, 3))

        self.image_standradisation_multi = tf.image.per_image_standardization(self.standardize_image_holder)
        self.flipped_img_multi = tf.image.flip_left_right(self.flipped_image_holder)
        
        self.ch = tf.placeholder(tf.int32)
        self.cw = tf.placeholder(tf.int32)
        
        self.cropped_boundingbox = tf.image.crop_to_bounding_box(self.crop, self.ch, self.cw, RESIZE_FINAL, RESIZE_FINAL)
        
        self.images_multi = tf.placeholder(dtype=tf.float32, shape=(12, 227, 227, 3))
        self.image_batch_mult = tf.stack(self.images_multi)
    
    def run_stack(self, images_array):
    	return self._sess.run(self.image_batch_single, feed_dict={self.images_single: images_array})   
	    
    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})
        
    def decode_jpeg(self, image_data, look):
        if look == 'single':
            return self._sess.run(self.image_standradisation, #self._decode_jpeg,
                                   feed_dict={self._decode_jpeg_data: image_data})
                    
        elif look == 'multi':
            image = self._sess.run(self.crop, #self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
            crops = []
            h = image.shape[0]
            w = image.shape[1]
            hl = h - RESIZE_FINAL
            wl = w - RESIZE_FINAL

            crop = self._sess.run(self.crop_image, feed_dict={self._decode_jpeg_data: image_data})
            standardizeImage = self._sess.run(self.image_standradisation_multi, feed_dict={self.standardize_image_holder: crop})
            crops.append(standardizeImage)
            flippedImage = self._sess.run(self.flipped_img_multi, feed_dict={self.flipped_image_holder: crop})
            crops.append(flippedImage)

            corners = [ (0, 0), (0, wl), (hl, 0), (hl, wl), (int(hl/2), int(wl/2))]
            for corner in corners:
                ch, cw = corner
                cropped = self._sess.run(self.cropped_boundingbox, feed_dict={self._decode_jpeg_data: image_data, self.ch: ch, self.cw: cw})
                standardizeImage_2 = self._sess.run(self.image_standradisation_multi, feed_dict={self.standardize_image_holder: cropped})
                crops.append(standardizeImage_2)
                flippedImage_2 = self._sess.run(self.flipped_img_multi, feed_dict={self.flipped_image_holder: cropped})
                crops.append(flippedImage_2)

            return self._sess.run(self.image_batch_mult, feed_dict={self.images_multi: crops})
            
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
        
def make_multi_image_batch(filenames, coder, number_of_images):
    """Process a multi-image batch, each with a single-look
    Args:
    filenames: list of paths
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
    image_buffer: string, JPEG encoding of RGB image.
    """
    images = []
    coder.num_img = number_of_images
    for filename in filenames:
    	with tf.gfile.FastGFile(filename, 'rb') as f:
                image_data = f.read()
            # Convert any PNG to JPEG's for consistency.
            if _is_png(filename):
                print('Converting PNG to JPEG for %s' % filename)
                image_data = coder.png_to_jpeg(image_data)
        
            image = coder.decode_jpeg(image_data, 'single')
            images.append(image)
    image_batch = coder.run_stack(images) 
    
    return image_batch

def make_multi_crop_batch(filename, coder):
    """Process a single image file.
    Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
    image_buffer: string, JPEG encoding of RGB image.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    # Convert any PNG to JPEG's for consistency.
    if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)
    
    image = coder.decode_jpeg(image_data, 'multi')

    return image   

def face_detection_model(model_type, model_path):
    model_type_lc = model_type.lower()
    if model_type_lc == 'yolo_tiny':
        from yolodetect import PersonDetectorYOLOTiny
        return PersonDetectorYOLOTiny(model_path)
    elif model_type_lc == 'yolo_face':
        from yolodetect import FaceDetectorYOLO
        return FaceDetectorYOLO(model_path)
    elif model_type == 'dlib':
        from dlibdetect import FaceDetectorDlib
        return FaceDetectorDlib(model_path)
    return ObjectDetectorCascadeOpenCV(model_path)
