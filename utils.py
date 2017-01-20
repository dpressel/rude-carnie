from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from datetime import datetime
import cv2
import sys
import math
import time
from data import inputs, standardize_image
import numpy as np
import tensorflow as tf

RESIZE_AOI = 256
RESIZE_FINAL = 227
FACE_PAD = 50

class FaceDetector(object):

    def __init__(self, model_name, basename = 'frontal-face', tgtdir = '.'):
        self.tgtdir = tgtdir
        self.basename = basename
        self.face_cascade = cv2.CascadeClassifier(model_name)
    
    def run(self, image_file, min_height_dec = 20, min_width_dec = 20, min_height_thresh=50, min_width_thresh=50):
        print(image_file)
        img = cv2.imread(image_file)
        min_h = int(max(img.shape[0] / min_height_dec, min_height_thresh))
        min_w = int(max(img.shape[1] / min_width_dec, min_width_thresh))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, minNeighbors=5, minSize=(min_h,min_w))

        images = []
        for i, (x,y,w,h) in enumerate(faces):
            images.append(self.sub_image('%s/%s-%d.jpg' % (self.tgtdir, self.basename, i+1), img, x, y, w, h))
        
        print('%d faces detected' % len(images))

        for (x,y,w,h) in faces:
            self.draw_rect(img, x, y, w, h)
            # Fix in case nothing found in the image
        outfile = '%s/%s.jpg' % (self.tgtdir, self.basename)
        cv2.imwrite(outfile, img)

        return images, outfile

    def sub_image(self, name, img, x, y, w, h):
        upper_cut = [min(img.shape[0], y+h+FACE_PAD), min(img.shape[1], x+w+FACE_PAD)]
        lower_cut = [max(y-FACE_PAD, 0), max(x-FACE_PAD, 0)]
        roi_color = img[lower_cut[0]:upper_cut[0], lower_cut[1]:upper_cut[1]]
        cv2.imwrite(name, roi_color)
        return name
        
    def draw_rect(self, img, x, y, w, h):
        upper_cut = [min(img.shape[0], y+h+FACE_PAD), min(img.shape[1], x+w+FACE_PAD)]
        lower_cut = [max(y-FACE_PAD, 0), max(x-FACE_PAD, 0)]
        cv2.rectangle(img, (lower_cut[1],lower_cut[0]),(upper_cut[1],upper_cut[0]), (255,0,0), 2)

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
