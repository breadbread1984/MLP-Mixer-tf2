#!/usr/bin/python3

from os.path import join, exists;
from absl import app, flags;
import tensorflow as tf;
from models import MLPMixer, configs;
from create_datasets import load_datasets;

FLAGS = flags.FLAGS;

def add_options():
  flags.DEFINE_integer('batch_size', 256, help = 'batch size');
  flags.DEFINE_enum('model', 'b16', enum_values = ['b16', 'b32', 'l16'], help = 'which mlp mixer is trained');
  flags.DEFINE_string('checkpoint', 'checkpoints', help = 'path to checkpoint');
  flags.DEFINE_integer('epochs', 25, help = 'epochs');
  flags.DEFINE_float('lr', 3e-4, help = 'learning rate');

def main(unused_argv):
  if exists(FLAGS.checkpoint):
    model = tf.keras.models.load_model(FLAGS.checkpoint, custom_objects = {'tf': tf}, compile = True);
    optimizer = model.optimizer;
  else:
    model = MLPMixer((32,32,3,), 10, configs[FLAGS.model]);
    optimizer = tf.keras.optimizers.Adam(FLAGS.lr);
    model.compile(optimizer = optimizer,
                  loss = [tf.keras.losses.SparseCategoricalCrossentropy()],
                  metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]);
  trainset, testset = load_datasets();
  trainset = trainset.shuffle(FLAGS.batch_size).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  testset = testset.shuffle(FLAGS.batch_size).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = FLAGS.checkpoint),
    tf.keras.callbacks.ModelCheckpoint(filepath = join(FLAGS.checkpoint, 'ckpt'), save_freq = 1000),
  ];
  model.fit(trainset, epochs = FLAGS.epochs, validation_data = testset, callbacks = callbacks);

if __name__ == "__main__":
  add_options();
  app.run(main);
