#for convenience of importing key world model functions to notebooks!

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
import tensorflow_probability as tfp

class MLPVAE(Model):
    def __init__(self, input_dim=348, z_size=32, kl_tolerance=0.5):
        super(MLPVAE, self).__init__()
        self.z_size = z_size
        self.kl_tolerance = kl_tolerance

        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(2 * z_size),  # output both mu and logvar
        ])

        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(z_size,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(input_dim, activation='linear'),  # output same shape as input
        ])

    def sample_z(self, mu, logvar):
        eps = tf.random.normal(shape=tf.shape(mu))
        sigma = tf.exp(0.5 * logvar)
        return mu + sigma * eps

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = tf.split(h, num_or_size_splits=2, axis=1)
        logvar = tf.clip_by_value(logvar, -10.0, 10.0)  # helps with exploding values
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        mu, logvar = self.encode(x)
        z = self.sample_z(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def compute_loss(self, x):
        x_recon, mu, logvar = self(x)
        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_recon), axis=1))
        kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
        kl_loss = tf.maximum(kl_loss, self.kl_tolerance * self.z_size)
        kl_loss = tf.reduce_mean(kl_loss)
        total_loss = recon_loss + kl_loss
        return total_loss, recon_loss, kl_loss
    
class MDNRNN(tf.keras.Model):
    def __init__(self, latent_dim, action_dim, hidden_dim=256, num_mixtures=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.input_dim = latent_dim + action_dim
        self.hidden_dim = hidden_dim
        self.num_mixtures = num_mixtures

        # LSTM
        self.lstm = layers.LSTM(hidden_dim, return_sequences=True, return_state=True)

        # MDN output: means, stddevs, and mixture weights for latent prediction
        self.mdn_dense = layers.Dense(num_mixtures * (2 * latent_dim + 1))

        # Predict reward (scalar)
        self.reward_dense = layers.Dense(1)

        # Predict done (binary classification)
        self.done_dense = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, initial_state=None, training=False):
        """
        inputs: (batch, seq_len, latent_dim + action_dim)
        """
        lstm_out, h, c = self.lstm(inputs, initial_state=initial_state, training=training)

        mdn_out = self.mdn_dense(lstm_out)
        reward_pred = self.reward_dense(lstm_out)
        done_pred = self.done_dense(lstm_out)

        return mdn_out, reward_pred, done_pred, [h, c]

    def get_mdn_params(self, mdn_out):
        """Split MDN output into pi, mu, sigma."""
        out = tf.reshape(mdn_out, [-1, self.num_mixtures, 2 * self.latent_dim + 1])
        pi = out[:, :, 0]
        mu = out[:, :, 1 : 1 + self.latent_dim]
        log_sigma = out[:, :, 1 + self.latent_dim :]
        sigma = tf.exp(log_sigma)

        pi = tf.nn.softmax(pi, axis=-1)  # mixture weights
        return pi, mu, sigma
    
class LinearController(Model):
    def __init__(self, input_dim, action_dim):
        super(LinearController, self).__init__()
        self.linear = layers.Dense(action_dim, activation='tanh', use_bias=False)
        self.linear(tf.zeros((1, input_dim)))  # force build so weights are initialized

    def call(self, inputs):
        return self.linear(inputs)

    def get_weights_flat(self):
        return tf.reshape(self.linear.kernel, [-1])

    def set_weights_flat(self, flat_weights):
        new_weights = tf.reshape(flat_weights, self.linear.kernel.shape)
        self.linear.kernel.assign(new_weights)

def make_world_model_policy(vae, mdn_rnn, controller):
    h = tf.zeros((1, 256))
    state = [h, h]

    def policy(obs):
        nonlocal state, h
        x = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)
        mu, _ = vae.encode(x)
        z = mu

        zh = tf.concat([z, h], axis=1)
        action = controller(zh).numpy()[0]

        rnn_input = tf.concat([z, action[None]], axis=1)
        rnn_input = tf.expand_dims(rnn_input, axis=1)  # shape: (1, 1, 50)
        _, _, _, state = mdn_rnn(rnn_input, initial_state=state)
        h = state[0]

        return action

    return policy