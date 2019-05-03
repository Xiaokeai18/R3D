# Coder: Wang Pei
# Github: https://github.com/xiaokeai18/R3D
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
import os
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import resnet
import math
import numpy as np

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
#flags.DEFINE_float('learning_rate', 0.0, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 15000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 10, 'Batch size.')
FLAGS = flags.FLAGS
MOVING_AVERAGE_DECAY = 0.9999
model_save_dir = './models'

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         FLAGS.NUM_FRAMES_PER_CLIP,
                                                         FLAGS.CROP_SIZE-20,
                                                         FLAGS.CROP_SIZE,
                                                         FLAGS.CHANNELS))
  labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
  return images_placeholder, labels_placeholder

def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def tower_loss(name_scope, logit, labels):
  cross_entropy_mean = tf.reduce_mean(
                  tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logit)
                  )
  tf.summary.scalar(
                  name_scope + '_cross_entropy',
                  cross_entropy_mean
                  )
                  
  tf.summary.scalar(name_scope + '_weight_decay_loss', tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) )

  # Calculate the total loss for the current tower.
  total_loss = cross_entropy_mean + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  tf.summary.scalar(name_scope + '_total_loss', tf.reduce_mean(total_loss) )
  return total_loss

def tower_acc(logit, labels):
  correct_pred = tf.equal(tf.argmax(logit, 1), labels)
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  return accuracy

def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def run_training():
  # Get the sets of images and labels for training, validation, and
  # Tell TensorFlow that the model will be built into the default Graph.

  # Create model directory
  if not os.path.exists(model_save_dir):
      os.makedirs(model_save_dir)
  use_pretrained_model = False 
  use_ckpt = False
  model_filename = "./models/r3d_model-7999"

  with tf.Graph().as_default():
    global_step = tf.get_variable(
                    'global_step',
                    [],
                    initializer=tf.constant_initializer(0),
                    trainable=False
                    )
    images_placeholder, labels_placeholder = placeholder_inputs(
                    FLAGS.batch_size * gpu_num
                    )
    tower_grads1 = []
    tower_grads2 = []
    logits = []
    opt_stable = tf.train.AdamOptimizer(1e-4)
    opt_finetuning = tf.train.AdamOptimizer(1e-3)

    for gpu_index in range(0, gpu_num):
      with tf.device('/gpu:%d' % gpu_index):
        
        
        logit = resnet.inference(
                        images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:,:,:,:],
                        5,
                        False
                        )
        loss_name_scope = ('gpud_%d_loss' % gpu_index)
        loss = tower_loss(
                        loss_name_scope,
                        logit,
                        labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size]
                        )
        grads1 = opt_stable.compute_gradients(loss)
        grads2 = opt_finetuning.compute_gradients(loss)
        tower_grads1.append(grads1)
        tower_grads2.append(grads2)
        logits.append(logit)
    logits = tf.concat(logits,0)
    accuracy = tower_acc(logits, labels_placeholder)
    tf.summary.scalar('accuracy', accuracy)
    grads1 = average_gradients(tower_grads1)
    grads2 = average_gradients(tower_grads2)
    apply_gradient_op1 = opt_stable.apply_gradients(grads1)
    apply_gradient_op2 = opt_finetuning.apply_gradients(grads2, global_step=global_step)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    train_op = tf.group(apply_gradient_op1, apply_gradient_op2, variables_averages_op)
    null_op = tf.no_op()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()

    # Create a session for running Ops on the Graph.
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True

    sess = tf.Session(
                    #config=tf.ConfigProto(allow_soft_placement=True)
                    config=config
                    )
    sess.run(init)
    if os.path.isfile(model_filename) and use_pretrained_model:
      saver.restore(sess, model_filename)
      print("Successfully load the pretrained data")
    if use_ckpt:
      ckpt = tf.train.get_checkpoint_state(model_save_dir)
      if ckpt and ckpt.model_checkpoint_path:
          saver.restore(sess, ckpt.model_checkpoint_path)
          print("Model restored...")

    # Create summary writter
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./visual_logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('./visual_logs/test', sess.graph)
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      train_images, train_labels, _, _, _ = input_data.read_clip_and_label(
                      filename='train.list',
                      batch_size=FLAGS.batch_size * gpu_num,
                      num_frames_per_clip=FLAGS.NUM_FRAMES_PER_CLIP,
                      crop_size=FLAGS.CROP_SIZE,
                      shuffle=True
                      )
      sess.run(train_op, feed_dict={
                      images_placeholder: train_images,
                      labels_placeholder: train_labels
                      })
      duration = time.time() - start_time
      print('Step %d: %.3f sec' % (step, duration))

      # Save a checkpoint and evaluate the model periodically.
      if (step) % 10 == 0 or (step + 1) == FLAGS.max_steps:
        saver.save(sess, os.path.join(model_save_dir, 'r3d_model'), global_step=step)
        print('Training Data Eval:')
        summary, acc = sess.run(
                        [merged, accuracy],
                        feed_dict={images_placeholder: train_images,
                            labels_placeholder: train_labels
                            })
        print ("accuracy: " + "{:.5f}".format(acc))
        train_writer.add_summary(summary, step)
        print('Validation Data Eval:')
        val_images, val_labels, _, _, _ = input_data.read_clip_and_label(
                        filename='test.list',
                        batch_size=FLAGS.batch_size * gpu_num,
                        num_frames_per_clip=FLAGS.NUM_FRAMES_PER_CLIP,
                        crop_size=FLAGS.CROP_SIZE,
                        shuffle=True
                        )
        summary, acc = sess.run(
                        [merged, accuracy],
                        feed_dict={
                                        images_placeholder: val_images,
                                        labels_placeholder: val_labels
                                        })
        print ("accuracy: " + "{:.5f}".format(acc))
        test_writer.add_summary(summary, step)
  print("done")

def main(_):
  run_training()

if __name__ == '__main__':
  tf.app.run()
