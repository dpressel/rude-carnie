from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
from data import inputs
import numpy as np
import tensorflow as tf
from model import select_model, get_checkpoint
from utils import ImageCoder, make_batch, FaceDetector
import os
import json
RESIZE_FINAL = 227
GENDER_LIST =['M','F']
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']

tf.app.flags.DEFINE_string('model_dir', '',
                           'Model directory (where training data lives)')

tf.app.flags.DEFINE_string('class_type', 'age',
                           'Classification type (age|gender)')


tf.app.flags.DEFINE_string('device_id', '/cpu:0',
                           'What processing unit to execute inference on')

tf.app.flags.DEFINE_string('filename', '',
                           'File to processs')

tf.app.flags.DEFINE_string('checkpoint', 'checkpoint',
                          'Checkpoint basename')

tf.app.flags.DEFINE_string('model_type', 'default',
                           'Type of convnet')

tf.app.flags.DEFINE_string('requested_step', '', 'Within the model directory, a requested step to restore e.g., 9000')

tf.app.flags.DEFINE_boolean('single_look', False, 'single look at the image or multiple crops')

tf.app.flags.DEFINE_string('face_detection_model', '', 'Do frontal face detection with model specified')

FLAGS = tf.app.flags.FLAGS


def classify(sess, label_list, softmax_output, coder, images, image_file):

    print('Running file %s' % image_file)
    image_batch = make_batch(image_file, coder, not FLAGS.single_look)
    
    print(image_batch)
    batch_results = sess.run(softmax_output, feed_dict={images:image_batch.eval()})
    output = batch_results[0]
    batch_sz = batch_results.shape[0]
    for i in range(1, batch_sz):
        output = output + batch_results[i]
        
    output /= batch_sz
    best = np.argmax(output)
        
    print('Guess @ 1 %s, prob = %.2f' % (label_list[best], output[best]))
    
    nlabels = len(label_list)
    if nlabels > 2:
        output[best] = 0
        second_best = np.argmax(output)

        print('Guess @ 2 %s, prob = %.2f' % (label_list[second_best], output[second_best]))

         
def main(argv=None):  # pylint: disable=unused-argument


    with tf.Session() as sess:

        label_list = AGE_LIST if FLAGS.class_type == 'age' else GENDER_LIST
        nlabels = len(label_list)

        print('Executing on %s' % FLAGS.device_id)
        model_fn = select_model(FLAGS.model_type)

        images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
        logits = model_fn(nlabels, images, 1, False)
        init = tf.initialize_all_variables()
            
        requested_step = FLAGS.requested_step if FLAGS.requested_step else None
        
        checkpoint_path = '%s' % (FLAGS.model_dir)

        model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, FLAGS.checkpoint)
            
        saver = tf.train.Saver()
        saver.restore(sess, model_checkpoint_path)
                        
        softmax_output = tf.nn.softmax(logits)

        coder = ImageCoder()

        files = []

        if FLAGS.face_detection_model:
            print('Using face detector %s' % FLAGS.face_detection_model)
            face_detect = FaceDetector(FLAGS.face_detection_model)
            face_files, rectangles = face_detect.run(FLAGS.filename)
            files += face_files

        if len(files) == 0:
            files.append(FLAGS.filename)


        for f in files:
            classify(sess, label_list, softmax_output, coder, images, f)

        
if __name__ == '__main__':
    tf.app.run()
