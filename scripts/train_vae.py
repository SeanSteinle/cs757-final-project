import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model

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
    
def create_dataset(x_train, batch_size=64, shuffle_buffer=10000):
    # Assuming x_train is a NumPy array of shape [n_samples, 348]
    dataset = tf.data.Dataset.from_tensor_slices(x_train.astype(np.float32))
    dataset = dataset.shuffle(shuffle_buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def train_vae(model, dataset, epochs=10, learning_rate=1e-4):
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    for epoch in range(epochs):
        total_loss = 0.0
        total_batches = 0
        for x_batch in dataset:
            with tf.GradientTape() as tape:
                loss, recon_loss, kl_loss = model.compute_loss(x_batch)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            total_loss += loss.numpy()
            total_batches += 1

        avg_loss = total_loss / total_batches
        print("Epoch {}: avg loss = {:.4f}".format(epoch+1, avg_loss))

if __name__ == "__main__":
    #parameters
    experiment_id = "Humanoid-v5_500000"
    save_path = f"../models/vae/{experiment_id}.weights.h5"
    observations_path = f"../data/processed/{experiment_id}/observations.npy"
    Z_DIM = 32
    VAE_EPOCHS = 25

    #load data
    x_train = np.load(observations_path)
    X_DIM = x_train.shape[1]

    #standardize data and create dataset
    x_train = (x_train - np.mean(x_train, axis=0)) / (np.std(x_train, axis=0) + 1e-6)
    dataset = create_dataset(x_train, batch_size=64)

    #instantiate and train model
    vae = MLPVAE(input_dim=X_DIM, z_size=Z_DIM)
    train_vae(vae, dataset, epochs=VAE_EPOCHS)

    #save out model
    vae.save_weights(save_path)