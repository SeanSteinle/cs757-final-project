# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import tensorflow_probability as tfp
from train_vae import MLPVAE

#CORE MODEL CLASS
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

#LOSS IMPLEMENTATION
def mdn_loss(z_target, pi, mu, sigma, eps=1e-8):
    """
    z_target: [batch * seq_len, latent_dim]
    pi: [batch * seq_len, num_mixtures]
    mu: [batch * seq_len, num_mixtures, latent_dim]
    sigma: [batch * seq_len, num_mixtures, latent_dim]
    """
    # Expand target for broadcasting: [batch, 1, latent_dim]
    z_expanded = tf.expand_dims(z_target, axis=1)

    # Create component Gaussians
    normal_dist = tfp.distributions.Normal(loc=mu, scale=sigma)
    log_probs = normal_dist.log_prob(z_expanded)  # shape: [batch, num_mixtures, latent_dim]

    # Sum over latent_dim: total log prob of each mixture component
    log_probs = tf.reduce_sum(log_probs, axis=-1)  # shape: [batch, num_mixtures]

    # Weight by mixture coefficients
    weighted_log_probs = log_probs + tf.math.log(pi + eps)  # log(pi * P)
    
    # LogSumExp over mixture components to marginalize
    log_likelihood = tf.reduce_logsumexp(weighted_log_probs, axis=-1)  # shape: [batch]

    # Negative log-likelihood
    return -tf.reduce_mean(log_likelihood)

def combined_loss(z_target, pi, mu, sigma, reward_target, reward_pred, done_target, done_pred,
                  reward_weight=1.0, done_weight=1.0):
    loss_mdn = mdn_loss(z_target, pi, mu, sigma)
    loss_reward = tf.reduce_mean(tf.square(reward_target - reward_pred))
    loss_done = tf.reduce_mean(tf.keras.losses.binary_crossentropy(done_target, done_pred))

    return loss_mdn + reward_weight * loss_reward + done_weight * loss_done

#TRAINING FUNCTION
def train_mdnrnn(model, dataset, epochs=10, learning_rate=1e-4, reward_weight=1.0, done_weight=1.0):
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_mdn_loss = 0.0
        total_reward_loss = 0.0
        total_done_loss = 0.0
        total_batches = 0

        for z_action, (z_next, reward, done) in dataset:
            # Flatten z_action into shape (batch, seq_len, latent_dim + action_dim)
            with tf.GradientTape() as tape:
                # Forward pass through the model
                mdn_out, reward_pred, done_pred, _ = model(z_action, training=True)
                pi, mu, sigma = model.get_mdn_params(mdn_out)

                # Compute MDN, reward, and done losses
                loss_mdn = mdn_loss(tf.reshape(z_next, [-1, model.latent_dim]), pi, mu, sigma)
                loss_reward = tf.reduce_mean(tf.square(reward - reward_pred))
                loss_done = tf.reduce_mean(tf.keras.losses.binary_crossentropy(done, done_pred))

                # Total loss
                total_loss = loss_mdn + reward_weight * loss_reward + done_weight * loss_done

            # Compute gradients and apply
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Accumulate the loss values
            total_loss += total_loss.numpy()
            total_mdn_loss += loss_mdn.numpy()
            total_reward_loss += loss_reward.numpy()
            total_done_loss += loss_done.numpy()

            total_batches += 1

        # Compute and print average losses for the epoch
        avg_loss = total_loss / total_batches
        avg_mdn_loss = total_mdn_loss / total_batches
        avg_reward_loss = total_reward_loss / total_batches
        avg_done_loss = total_done_loss / total_batches

    print("Epoch {}:".format(epoch + 1))
    print("  avg loss = {:.4f}, mdn_loss = {:.4f}, reward_loss = {:.4f}, done_loss = {:.4f}".format(
        avg_loss, avg_mdn_loss, avg_reward_loss, avg_done_loss))

#DATA HANDLING
def preprocess_data(observations, actions, rewards, dones, sequence_length):
    """
    Yields tuples of the form:
    z_action: (sequence_length, latent_dim + action_dim)
    targets:  (z_next, reward, done), each of shape (sequence_length, ...)
    """
    for i in range(len(observations) - sequence_length):
        z_seq = observations[i:i+sequence_length]
        a_seq = actions[i:i+sequence_length]
        z_action = np.concatenate([z_seq, a_seq], axis=-1).astype(np.float32)

        z_next = observations[i+1:i+sequence_length+1].astype(np.float32)
        reward = rewards[i+1:i+sequence_length+1].astype(np.float32)
        done = dones[i+1:i+sequence_length+1].astype(np.float32)

        yield (
            z_action.astype(np.float32),
            (
                z_next.astype(np.float32),
                reward[:, None].astype(np.float32),  # <–– reshape from (T,) to (T,1)
                done[:, None].astype(np.float32)     # <–– reshape from (T,) to (T,1)
            )
        )

if __name__ == "__main__":
    #parameters for extracting z values
    experiment_id = "Humanoid-v5_500000"
    obs_path = "../data/processed/{}/observations.npy".format(experiment_id)
    vae_path = "../models/vae/{}.weights.h5".format(experiment_id)
    z_path = "../data/processed/{}/z.npy".format(experiment_id)
    mdnrnn_path = "../models/mdnrnn/{}.weights.h5".format(experiment_id)
    Z_DIM = 32

    #first, need to load our VAE and update our dataset with Z predictions for our MDN-RNN to train on.
    observations = np.load(obs_path)
    X_DIM = observations.shape[1]
    vae = MLPVAE(input_dim=X_DIM, z_size=Z_DIM) #instantiate new model object 
    vae(tf.zeros((1, X_DIM))) #invoke it to build its shape 
    vae.load_weights(vae_path) #now load weights into empty vector
    mu, logvar = vae.encode(observations)
    z = vae.sample_z(mu, logvar).numpy()
    np.save(z_path, z)

    #parameters for MDN-RNN training
    actions = np.load('../data/processed/{}/actions.npy'.format(experiment_id))
    rewards = np.load('../data/processed/{}/rewards.npy'.format(experiment_id))
    done = np.load('../data/processed/{}/done.npy'.format(experiment_id))
    SEQ_LEN = 10 #number of samples to process at a time, 'context' of RNN
    A_DIM = actions.shape[1] #dimensionality of action space
    batch_size = 32
    EPOCHS = 100

    train_dataset = tf.data.Dataset.from_generator(
        lambda: preprocess_data(z, actions, rewards, done, SEQ_LEN),
        output_signature=(
            tf.TensorSpec(shape=(SEQ_LEN, Z_DIM + A_DIM), dtype=tf.float32),
            (
                tf.TensorSpec(shape=(SEQ_LEN, Z_DIM), dtype=tf.float32),
                tf.TensorSpec(shape=(SEQ_LEN, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(SEQ_LEN, 1), dtype=tf.float32),
            )
        )
    )

    train_dataset = train_dataset.batch(batch_size).shuffle(1000)
    mdnrnn = MDNRNN(latent_dim=Z_DIM, action_dim=A_DIM)
    train_mdnrnn(mdnrnn, train_dataset, EPOCHS)
    mdnrnn.save_weights('../models/mdnrnn/{}_{}.weights.h5'.format(experiment_id, EPOCHS)) #save ONLY weights -- much simpler than serializing the entire object