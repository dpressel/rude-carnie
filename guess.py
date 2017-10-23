from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from model import select_model, get_checkpoint
from utils import *
import os
import csv
import cv2
import dlib

RESIZE_FINAL = 227
GENDER_LIST = ['M', 'F']
AGE_LIST = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
MAX_BATCH_SZ = 128

tf.app.flags.DEFINE_string('model_dir', '',
                           'Model directory (where age model lives)')

tf.app.flags.DEFINE_string('model_sex_dir', '',
                           'Model directory (where sex model lives)')

tf.app.flags.DEFINE_string('class_type', 'age',
                           'Classification type (age|gender)')

tf.app.flags.DEFINE_string('device_id', '/cpu:0',
                           'What processing unit to execute inference on')

tf.app.flags.DEFINE_string('filename', '',
                           'File (Image) or File list (Text/No header TSV) to process')

tf.app.flags.DEFINE_string('input', 'video', "input type (video|photo)")

tf.app.flags.DEFINE_string('target', '',
                           'CSV file containing the filename processed along with best guess and score')

tf.app.flags.DEFINE_string('checkpoint', 'checkpoint',
                           'Checkpoint basename')

tf.app.flags.DEFINE_string('model_type', "inception",
                           'Type of convnet')

tf.app.flags.DEFINE_string('requested_step', '', 'Within the model directory, a requested step to restore e.g., 9000')

tf.app.flags.DEFINE_boolean('single_look', False, 'single look at the image or multiple crops')

tf.app.flags.DEFINE_string('face_detection_model', '', 'Do frontal face detection with model specified')

tf.app.flags.DEFINE_string('face_detection_type', 'dlib', 'Face detection model type (yolo_tiny|cascade)')

FLAGS = tf.app.flags.FLAGS


def one_of(fname, types):
    return any([fname.endswith('.' + ty) for ty in types])


def resolve_file(fname):
    if os.path.exists(fname):
        return fname
    for suffix in ('.jpg', '.png', '.JPG', '.PNG', '.jpeg'):
        cand = fname + suffix
        if os.path.exists(cand):
            return cand
    return None


def classify_many_single_crop(cl_model, label_list, coder, image_files, writer):
    try:
        num_batches = math.ceil(len(image_files) / MAX_BATCH_SZ)
        pg = ProgressBar(num_batches)
        for j in range(num_batches):
            start_offset = j * MAX_BATCH_SZ
            end_offset = min((j + 1) * MAX_BATCH_SZ, len(image_files))

            batch_image_files = image_files[start_offset:end_offset]
            print(start_offset, end_offset, len(batch_image_files))
            image_batch = make_multi_image_batch(batch_image_files, coder)
            batch_results = cl_model.run(image_batch)
            batch_sz = batch_results.shape[0]

            for i in range(batch_sz):
                output_i = batch_results[i]
                best_i = np.argmax(output_i)
                best_choice = (label_list[best_i], output_i[best_i])
                print('Guess @ 1 %s, prob = %.2f' % best_choice)
                if writer is not None:
                    f = batch_image_files[i]
                    writer.writerow((f, best_choice[0], '%.2f' % best_choice[1]))
            pg.update()
        pg.done()
    except Exception as e:
        print(e)
        print('Failed to run all images')


def classify(cl_model, label_list, coder, image_file, writer, c_type):
    try:
        print('Running file %s' % image_file)
        image_batch = make_multi_crop_batch(image_file, coder)
        batch_results = cl_model.run(image_batch)
        output = batch_results[0]
        batch_sz = batch_results.shape[0]

        for i in range(1, batch_sz):
            output = output + batch_results[i]

        output /= batch_sz
        best = np.argmax(output)
        best_choice = (label_list[best], output[best])
        print('Guess @ 1 %s, prob = %.2f' % best_choice)
        nlabels = len(label_list)
        if nlabels > 2:
            output[best] = 0
            second_best = np.argmax(output)
            print('Guess @ 2 %s, prob = %.2f' % (label_list[second_best], output[second_best]))

        if writer is not None:
            writer.writerow((image_file, best_choice[0], '%.2f' % best_choice[1]))
        if c_type == 1:
            return "age: %s prob: %.2f" % best_choice
        else:
            return "sex: %s prob: %.2f" % best_choice

    except Exception as e:
        print(e)
        print('Failed to run image %s ' % image_file)


def list_images(srcfile):
    with open(srcfile, 'r') as csvfile:
        delim = ',' if srcfile.endswith('.csv') else '\t'
        reader = csv.reader(csvfile, delimiter=delim)
        if srcfile.endswith('.csv') or srcfile.endswith('.tsv'):
            print('skipping header')
            _ = next(reader)

        return [row[0] for row in reader]


def run():
    files = []
    if FLAGS.face_detection_model:
        print('Using face detector (%s) %s' % (FLAGS.face_detection_type, FLAGS.face_detection_model))
        face_detect = dlib.get_frontal_face_detector()

    model_age = ImportGraph(FLAGS.model_dir, "age")
    model_sex = ImportGraph(FLAGS.model_sex_dir, "sex")

    coder = ImageCoder()

    # Support a batch mode if FLAGS.filename is a dir
    if os.path.isdir(FLAGS.filename):
        for relpath in os.listdir(FLAGS.filename):
            abspath = os.path.join(FLAGS.filename, relpath)

            if os.path.isfile(abspath) and any(
                    [abspath.endswith('.' + ty) for ty in ('jpg', 'png', 'JPG', 'PNG', 'jpeg')]):
                print(abspath)
                files.append(abspath)
    else:
        files.append(FLAGS.filename)
        # If it happens to be a list file, read the list and clobber the files
        if any([FLAGS.filename.endswith('.' + ty) for ty in ('csv', 'tsv', 'txt')]):
            files = list_images(FLAGS.filename)

    writer = None
    output = None
    if FLAGS.target:
        print('Creating output file %s' % FLAGS.target)
        output = open(FLAGS.target, 'w')
        writer = csv.writer(output)
        writer.writerow(('file', 'label', 'score'))

    image_files = list(filter(lambda x: x is not None, [resolve_file(f) for f in files]))
    print(image_files)

    if FLAGS.single_look:
        classify_many_single_crop(model_age, AGE_LIST, coder, image_files, writer)
    else:
        for image_file in image_files:
            if FLAGS.face_detection_model:
                frame = cv2.imread(image_file)
                frame_classify(frame, face_detect, model_age, model_sex, coder, writer)

    if output is not None:
        output.close()


class ImportGraph(object):

    def __init__(self, model_dir, class_type):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # import saved model from loc into local graph
            label_list = AGE_LIST if class_type == 'age' else GENDER_LIST
            nlabels = len(label_list)
            model_fn = select_model(FLAGS.model_type)
            self.images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
            logits = model_fn(nlabels, self.images, 1, False)
            saver = tf.train.Saver()
            requested_step = FLAGS.requested_step if FLAGS.requested_step else None
            checkpoint_path = '%s' % model_dir
            model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, FLAGS.checkpoint)
            saver.restore(self.sess, model_checkpoint_path)
            self.softmax_output = tf.nn.softmax(logits)

    def run(self, data):
        with tf.Session().as_default():
            data = data.eval()
        return self.sess.run(self.softmax_output, feed_dict={self.images: data})


def frame_classify(frame, face_detect, model_age, model_sex, coder, writer):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rectangles = face_detect(gray, 1)
    height, width, _ = frame.shape
    for rectangle in rectangles:
        add_h = int(0.4 * (rectangle.bottom() - rectangle.top()))
        add_w = int(0.4 * (rectangle.right() - rectangle.left()))
        top = rectangle.top() - add_h if (rectangle.top() - add_h) > 0 else 0
        bottom = rectangle.bottom()+add_h if (rectangle.bottom()+add_h) < height else height
        left = rectangle.left()-add_w if (rectangle.left()-add_w) > 0 else 0
        right = rectangle.right()+add_w if (rectangle.right()+add_w) < width else width
        crop_img = frame[top:bottom, left:right]
        face_file = './cro.jpg'
        cv2.imwrite(face_file, crop_img)
        cv2.rectangle(frame, (rectangle.left(), rectangle.top()),
                      (rectangle.right(), rectangle.bottom()), (0, 0, 255), 1)
        age_label = classify(model_age, AGE_LIST, coder, face_file, writer, 1)
        sex_label = classify(model_sex, GENDER_LIST, coder, face_file, writer, 2)

        cv2.putText(frame, age_label, (rectangle.left(), rectangle.bottom() + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, sex_label, (rectangle.left(), rectangle.bottom() + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow("detect", frame)
    cv2.waitKey(1)


def run_camera():
    if FLAGS.face_detection_model:
        print('Using face detector (%s) %s' % (FLAGS.face_detection_type, FLAGS.face_detection_model))
        face_detect = dlib.get_frontal_face_detector()

    model_age = ImportGraph(FLAGS.model_dir, "age")
    model_sex = ImportGraph(FLAGS.model_sex_dir, "sex")

    coder = ImageCoder()
    camera = cv2.VideoCapture(0)
    count = 0
    writer = None
    cv2.namedWindow('detect', cv2.WINDOW_NORMAL)

    while camera.isOpened():
        count += 1
        ret, frame = camera.read()
        if not ret:
            print('camera error')
            break
        if count > 30:
            count = 0
            cv2.imwrite(FLAGS.filename, frame)
            if FLAGS.face_detection_model:
                frame_classify(frame, face_detect, model_age, model_sex, coder, writer)


def main(_):  # pylint: disable=unused-argument
    if FLAGS.input == "video":
        run_camera()
    else:
        run()


if __name__ == '__main__':
    tf.app.run()
