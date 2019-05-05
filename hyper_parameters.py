# Coder: Wang Pei
# Github: https://github.com/xiaokeai18/
# ==============================================================================
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_integer('num_residual_blocks', 5, '''How many residual blocks do you want''')
tf.app.flags.DEFINE_float('weight_decay', 0.0001, '''scale for l2 regularization''')
tf.app.flags.DEFINE_string('version', 'test_1', '''A version number defining the directory to save
logs and checkpoints''')
tf.app.flags.DEFINE_integer('CROP_SIZE', 80, '''SIZE of the cutted image prepared to train & test''')
tf.app.flags.DEFINE_integer('NUM_FRAMES_PER_CLIP', 24, '''number of frames per sample video for input''')
tf.app.flags.DEFINE_integer('CHANNELS', 3, '''number of channel of the input image''')

tf.app.flags.DEFINE_integer('NUM_CLASSES', 5, '''number of classes''')
train_dir = 'logs_' + FLAGS.version + '/'
