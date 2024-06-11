import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.optimizers import Adam
from DeepEnv import DeepEnv
from model import BalloonGameSolo

class ReinforceAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        # Assurez-vous de passer les arguments nécessaires à la méthode build_model
        self.model = self.build_model(state_size, action_size)

    def build_model(self, state_size, action_size):
        model = Sequential([
            Dense(128, activation='relu', input_dim=state_size),
            Dense(64, activation='relu'),
            Dense(action_size, activation='softmax')  # La dernière couche avec activation 'softmax'
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model
    
    def available_actions_mask(self):
        num_dice_in_play = self.get_rolls()  # Utiliser la méthode existante pour obtenir le nombre de dés en jeu
        mask = [1.0 if i < num_dice_in_play else 0.0 for i in range(self.ACTION_SIZE)]
        return mask


    # La méthode act reste inchangée
    def act(self, state, env):
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)

        policy = self.model.predict(state)[0]

        # Obtenir le masque d'actions disponibles de l'environnement
        action_mask = env.available_actions_mask()
        # Appliquer le masque sur les probabilités d'actions
        masked_policy = policy * action_mask
        # Normaliser les probabilités après l'application du masque
        masked_policy /= np.sum(masked_policy)

        return np.random.choice(self.action_size, p=masked_policy)


    
    def train(self, states, actions, rewards):
        actions_one_hot = tf.keras.utils.to_categorical(actions, self.action_size)
        discounted_rewards = self.discount_rewards(rewards)
        self.model.train_on_batch(states, actions_one_hot, sample_weight=discounted_rewards)
    
    def discount_rewards(self, rewards, gamma=0.99):
        discounted_r = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * gamma + rewards[t]
            discounted_r[t] = running_add
        discounted_r -= np.mean(discounted_r)
        discounted_r /= (np.std(discounted_r) + 1e-8)
        return discounted_r

env = DeepEnv(BalloonGameSolo())
agent = ReinforceAgent(env.OBS_SIZE, env.ACTION_SIZE)

def run_episode(env, agent):
    states, actions, rewards = [], [], []
    state = env.reset()
    done = False

    while not done:
        # Passer 'env' comme argument à 'act'
        action = agent.act(state, env)
        reroll_indices = env.get_reroll_indices(action)
        env.roll_dice(reroll_indices)
        
        next_state, reward, done = env.step()
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state

    return states, actions, rewards


def train_agent(episodes=1000):
    for episode in range(episodes):
        states, actions, rewards = run_episode(env, agent)
        agent.train(np.vstack(states), np.array(actions), np.array(rewards))

train_agent(episodes=1000)
