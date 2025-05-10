import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, saving
import gymnasium as gym
from train_vae import MLPVAE
from train_mdnrnn import MDNRNN

# --- Controller model ---
@saving.register_keras_serializable()
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

# --- Evolution strategy ---
def evolve_controller(controller, vae, mdn_rnn, env, generations=10, pop_size=64, sigma=0.1, elite_frac=0.2):
    input_dim = controller.linear.kernel.shape[0]
    action_dim = controller.linear.kernel.shape[1]
    weight_dim = input_dim * action_dim

    base_weights = controller.get_weights_flat().numpy()
    elite_num = max(1, int(pop_size * elite_frac))

    for gen in range(generations):
        population = [base_weights + sigma * np.random.randn(weight_dim) for _ in range(pop_size)]
        scores = []

        for i, individual in enumerate(population):
            controller.set_weights_flat(individual)
            reward = evaluate_controller(controller, vae, mdn_rnn, env)
            scores.append((reward, individual))

        scores.sort(key=lambda x: -x[0])
        elites = [w for _, w in scores[:elite_num]]
        new_mean = np.mean(elites, axis=0)
        base_weights = new_mean

        print("Gen {}: Best score = {}".format(gen+1, scores[0][0]))

    controller.set_weights_flat(base_weights)
    return controller

def evaluate_controller(controller, vae, mdn_rnn, env, max_steps=1000):
    total_reward = 0
    obs, _ = env.reset()
    done = False
    h = tf.zeros((1, 256))  # RNN hidden state (adjust as needed)

    state = [h, h]  # Initialize state for MDN-RNN (h, c)
    for step in range(max_steps):
        x = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)
        mu, _ = vae.encode(x)
        z = mu  # Use the mean (mu) as the latent vector

        zh = tf.concat([z, h], axis=1)  # Concatenate latent vector and hidden state
        action = controller(zh).numpy()[0]  # Get action from controller

        # Step in the environment
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Update hidden state using the MDN-RNN
        rnn_input = tf.concat([z, action[None]], axis=1)  # (latent_dim + action_dim)
        rnn_input = tf.expand_dims(rnn_input, axis=1)  # Shape becomes (1, 1, 50)

        mdn_out, reward_pred, done_pred, state = mdn_rnn(rnn_input, initial_state=state)

        total_reward += reward

        # Optionally: You can use reward_pred and done_pred for monitoring

        if done:
            break

    return total_reward

if __name__ == "__main__":
    #parameters
    experiment_id, MDNRNN_EPOCHS = 'Humanoid-v5_10000', 5
    O_DIM = 348
    Z_DIM = 32
    H_DIM = 256
    SEQ_LEN = 10
    env = gym.make("Humanoid-v5", render_mode=None)
    A_DIM = env.action_space.shape[0]
    CONTROLLER_GENS = 5

    #load models
    vae = MLPVAE(input_dim=O_DIM, z_size=Z_DIM) #instantiate new model object 
    vae(tf.zeros((1, O_DIM))) #invoke it to build its shape 
    vae.load_weights('../models/vae/{}.weights.h5'.format(experiment_id)) #now load weights into empty vector
    mdnrnn = MDNRNN(latent_dim=Z_DIM, action_dim=A_DIM) #instantiate new model object 
    mdnrnn(tf.zeros((1, SEQ_LEN, Z_DIM+A_DIM))) #invoke it to build its shape 
    mdnrnn.load_weights('../models/mdnrnn/{}_{}.weights.h5'.format(experiment_id, MDNRNN_EPOCHS)) #now load weights into empty vector

    controller = LinearController(input_dim=Z_DIM + H_DIM, action_dim=A_DIM)
    trained_controller = evolve_controller(controller, vae, mdnrnn, env, generations = CONTROLLER_GENS)
    controller.save_weights('../models/controller/{}_{}_{}.weights.h5'.format(experiment_id, MDNRNN_EPOCHS, CONTROLLER_GENS))