Rude Carnie: Age and Gender Deep Learning with TensorFlow
==========================================================

## Goal

Do face detection and age and gender classification on pictures

### Name

http://www.someecards.com/news/getting-old/howoldnet-takes-your-picture-and-uses-algorithms-to-guess-your-age-like-a-rude-carnie/

### Currently Supported Models

 - _Gil Levi and Tal Hassner, Age and Gender Classification Using Convolutional Neural Networks, IEEE Workshop on Analysis and Modeling of Faces and Gestures (AMFG), at the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), Boston, June 2015_

http://www.openu.ac.il/home/hassner/projects/cnn_agegender/
https://github.com/GilLevi/AgeGenderDeepLearning

 - Inception v3 with fine-tuning

### Running

There are several ways to use a pre-existing checkpoint to do age or gender classification.  By default, the code will simply assume that the image you provided has a face in it, and will run that image through a multi-pass classification using the corners and center.

  The --class_type parameter controls which task, and the --model_dir controls which checkpoint to restore.  There are advanced parameters for the checkpoint basename (--checkpoint) and the requested step number if there are multiple checkpoints in the directory (--requested_step)

Here is a run using Age classification on the latest checkpoint in a directory using 12-look (all corners + center + resized, along with flipped versions) averaging:

```
python2.7 guess.py --model_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/age_test_fold_is_1/run-20854 --filename /home/dpressel/Downloads/portraits/prince.jpg
```

You can also tell it to do a single image classification without the corners and center crop.  Here is a run using Age classification on the latest checkpoint in a directory, using a single look at the image

```
python2.7 guess.py --model_dir  /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/age_test_fold_is_1/run-20854 --filename /home/dpressel/Downloads/portraits/prince.jpg --single_look
```

Here is a version using gender, where we restore the checkpoint from a specific step:

```
python2.7 guess.py --model_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/gen_test_fold_is_0/run-31376 --class_type gender --requested_step 9999 --filename /home/dpressel/Downloads/portraits/prince.jpg 
```

Additionally, if you have an image with one or more frontal faces, you can run a face-detector upfront, and each detected face will be chipped out and run through classification individually:

```
python2.7 guess.py --model_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/gen_test_fold_is_0/run-31376 --class_type gender --requested_step 8000 --filename /home/dpressel/Downloads/portraits/halloween15.jpg --face_detection_model /usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml
```

### Training

You can use your own training data if you wish.  This is a little easier to do with gender, since there are many ways that you could come up with a training set for this, but it has been developed specifically with the Adience corpus in mind, and uses the pre-splits created by Levi and Hassner.

#### Download Adience data and folds

Get the folds, we dont need to run their preprocessing scripts since we are doing this in the preproc.py script using tensorflow

```
git clone https://github.com/GilLevi/AgeGenderDeepLearning
```

#### Pre-process data for training

First you will need to preprocess the data using preproc.py.  This assumes that there is a directory that is passed for an absolute directory, as well as a file containing a list of the training data images and the label itself, and the validation data, and test data if applicable.  The procproc.py program generates 'shards' for each of the datasets, each containing JPEG encoded RGB images of size 256x256

```
python2.7 preproc.py --fold_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/train_val_txt_files_per_fold/test_fold_is_0 --train_list age_train.txt --valid_list age_val.txt --data_dir /data/xdata/age-gender/aligned --output_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/age_test_fold_is_0

```

The training (etc) lists are expected in the --fold_dir, and they contain first the relative path from the --data_dir and second the numeric label:

```
dpressel@dpressel:~/dev/work/3csi-rd/dpressel/sh$ head /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/train_val_txt_files_per_fold/test_fold_is_0/age_train.txt 
10069023@N00/landmark_aligned_face.1924.10335948845_0d22490234_o.jpg 5
7464014@N04/landmark_aligned_face.961.10109081873_8060c8b0a5_o.jpg 4
28754132@N06/landmark_aligned_face.608.11546494564_2ec3e89568_o.jpg 2
10543088@N02/landmark_aligned_face.662.10044788254_2091a56ec3_o.jpg 3
66870968@N06/landmark_aligned_face.1227.11326221064_32114bf26a_o.jpg 4
7464014@N04/landmark_aligned_face.963.10142314254_8e96a97459_o.jpg 4
113525713@N07/landmark_aligned_face.1016.11784555666_8d43b6c493_o.jpg 3
30872264@N00/landmark_aligned_face.603.9575166089_f5f9cecc8c_o.jpg 5
10897942@N03/landmark_aligned_face.633.10372582914_382144ffe8_o.jpg 3
10792106@N03/landmark_aligned_face.522.11039121906_b047c90cc1_o.jpg 3
```

Gender is done much the same way:

```
python2.7 preproc.py --fold_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/train_val_txt_files_per_fold/test_fold_is_0 --train_list gender_train.txt --valid_list gender_val.txt --data_dir /data/xdata/age-gender/aligned --output_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/gen_test_fold_is_0
```

#### Train the model

Now that we have generated the training and validation shards, we can start training the program.  Here is a simple way to call the driver program to run for 10,000 iterations with a batch size of 128, and using SGD with momentum to train:

```
python2.7 train.py --train_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/age_test_fold_is_0 --max_steps 12000

```

You can easily monitor the job run by launching tensorboard with the --logdir specified in the program's initial output:

```
tensorboard --logdir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/gen_test_fold_is_0/run-31376

```
Then navigate to http://127.0.0.1:6006/ in your browser to see results.  The first tab (events) shows the loss over time, and the second shows the images that the network is seeing during training on batches.

** Evaluate the model **

The evaluation program is written to be run alongside the training or after the fact.  If you run it after the fact, you can specify a list of checkpoint steps to evaluate in sequence.  If you run while training is working, it will periodically rerun itself on the latest checkpoint.

Here is an example of running evaluation continuously.  The --run_id will live in the --train_dir (run-<id>) and is the product of a single run of training (the id is actually the PID used in training):

```
python2.7 eval.py  --run_id 15918 --train_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/gen_test_fold_is_0/ --eval_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/eval_gen_test_fold_is_0

```

Here is an after-the-fact run of eval that loops over the specified checkpoints and evaluates the performance on each:

```
python2.7 eval.py  --run_id 25079 --train_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/age_test_fold_is_0/ --eval_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/eval_age_test_fold_is_0 --requested_step_seq 7000,8000,9000,9999
```


