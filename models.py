#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_addons as tfa;

def MLPMixer(input_shape, patch_size, hidden_dim, num_blocks, tokens_mlp_dim, channels_mlp_dim, num_classes):
  inputs = tf.keras.Input(input_shape); # inputs.shape = (batch, h, w, c)
  results = tf.keras.layers.Conv2D(hidden_dim, kernel_size = (patch_size, patch_size), strides = (patch_size, patch_size))(inputs); # results.shape = (batch, h / patch, w / patch, hidden_dim)
  results = tf.keras.layers.Reshape((-1, results.shape[-1]))(results); # results.shape = (batch, h * w / patch ** 2, hidden_dim)
  for i in range(num_blocks):
    # 1) spatial mixing
    skip = results;
    results = tf.keras.layers.LayerNormalization()(results); # results.shape = (batch, h * w / patch ** 2, hidden_dim)
    results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,1)))(results); # results.shape = (batch, hidden_dim, h * w / patch ** 2)
    # Mixer Block
    channels = results.shape[-1];
    results = tf.keras.layers.Dense(tokens_mlp_dim, activation = tf.keras.activations.gelu)(results);
    results = tf.keras.layers.Dense(channels)(results); # results.shape = (batch, hidden_dim, h * w / patch ** 2)

    results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,1)))(results); # results.shape = (batch, h * w / patch ** 2, hidden_dim)
    results = tf.keras.layers.Add()([skip, results]);
    # 2) channel mixing
    skip = results;
    results = tf.keras.layers.LayerNormalization()(results);
    # Mixer Block
    channels = results.shape[-1];
    results = tf.keras.layers.Dense(channels_mlp_dim, activation = tf.keras.activations.gelu)(results);
    results = tf.keras.layers.Dense(channels)(results); # results.shape = (batch, h * w / patch ** 2, hidden_dim)
    
    results = tf.keras.layers.Add()([skip, results]);
  results = tf.keras.layers.LayerNormalization()(results);
  results = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis = 1))(results); # results.shape = (batch, hidden_dim)
  results = tf.keras.layers.Dense(num_classes)(results);  
  return tf.keras.Model(inputs = inputs, outputs = results);

if __name__ == "__main__":
  mlpmixer = MLPMixer((224,224,3), 16, 32, 4, 32, 32, 10);
  mlpmixer.save('mlpmixer.h5');