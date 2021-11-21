#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_addons as tfa;

configs = {
  'b16': {'patch_size': 16, 'hidden_dim': 768, 'num_blocks': 12, 'tokens_mlp_dim': 384, 'channels_mlp_dim': 3072},
  'b32': {'patch_size': 32, 'hidden_dim': 768, 'num_blocks': 12, 'tokens_mlp_dim': 384, 'channels_mlp_dim': 3072},
  'l16': {'patch_size': 16, 'hidden_dim': 1024, 'num_blocks': 24, 'tokens_mlp_dim': 512, 'channels_mlp_dim': 4096},
}

def MLPMixer(input_shape, num_classes, patch_size = 16, hidden_dim = 1024, num_blocks = 24, tokens_mlp_dim = 512, channels_mlp_dim = 4096):
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
  for key, config in configs.items():
    model = MLPMixer(input_shape = (224,224,3), num_classes = 1000, **config);
    model.save('%s.h5' % key);
  import numpy as np;
  inputs = np.random.normal(size = (4, 224,224,3));
  outputs = model(inputs);
  print(outputs.shape);
