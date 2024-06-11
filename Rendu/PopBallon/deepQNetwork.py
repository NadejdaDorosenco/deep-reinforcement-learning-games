from model import BalloonGameSolo
from DeepEnv import DeepEnv
import tensorflow as tf
import numpy as np
import time
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import legacy as legacy_optimizers
import pandas as pd

class DeepQNetwork(tf.keras.Model):

    def __init__(self, deep_env, epsilon=1, epsilon_decay=0.95, epsilon_min=0, gamma=0.95, learning_rate=0.001):
        super().__init__() 
        self.deep_env = deep_env 
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.model = self.build_dqn(learning_rate)
        self.nb_step = 0
        self.list_loss = []

    def reset(self):
        self.nb_step = 0
        self.list_loss = []
        self.deep_env.reset()
     
    def choose_action(self, state_vector):
        reshaped_state_vector = state_vector.reshape(1, -1)
        q_values = np.array(self.model(reshaped_state_vector)[0])

        action_mask = self.deep_env.available_actions_mask()
        masked_q_values = np.where(action_mask, q_values, float('-inf'))

        action = np.argmax(masked_q_values)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return action
    
    def step(self):
        self.deep_env.roll_dice()
        state_vector = self.deep_env.get_game_state_vector()
        while (5 - self.deep_env.get_rolls()) > 0:
            self.nb_step += 1

            action = self.choose_action(state_vector)
            if action == 0: 
                break
            elif 0 < action <= 15:
                reroll_indices = self.deep_env.get_reroll_indices(action)
                self.deep_env.roll_dice(reroll_indices) 
            else:
                print(f"Action en-dehors des actions possibles : {action}")
        reward, next_state, done = self.deep_env.step()
        loss = self.train(state_vector, action, reward, next_state, done)
        self.list_loss.append(loss)
        return next_state, reward, done
    
    def build_dqn(self, learning_rate=0.001):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.deep_env.OBS_SIZE,)),
            Dense(64, activation='relu'),
            Dense(self.deep_env.ACTION_SIZE, activation='linear')
        ])
        model.compile(optimizer=legacy_optimizers.Adam(learning_rate=learning_rate), loss='mse')
        return model
    
    @tf.function
    def train(self, state, action, reward, next_state, done):
        state = tf.expand_dims(state, axis=0)
        next_state = tf.expand_dims(next_state, axis=0)
        action = tf.convert_to_tensor([action], dtype=tf.int32)  
        batch_indices = tf.range(0, tf.shape(state)[0])
        indices = tf.stack([batch_indices, action], axis=1) 
        
        with tf.GradientTape() as tape:
            current_q_values = self.model(state, training=True)
            current_q = tf.gather_nd(current_q_values, indices)
            future_q_values = self.model(next_state, training=False)
            max_future_q = tf.reduce_max(future_q_values, axis=1)
            target_q = reward + (1.0 - done) * self.gamma * max_future_q
            loss = tf.reduce_mean(tf.square(target_q - current_q))
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss
    
def main():
    game_instance = BalloonGameSolo()
    env = DeepEnv(game_instance)    
    dqn_model = DeepQNetwork(env)
    start_time = time.time()  

    episodes = 1000
    scores_finaux = []
    losses = []
    steps = []
    list_reward = []

    for i in range(episodes):
        state = dqn_model.reset()
        total_reward = 0
        done = False
        while not done:
            next_state, reward, done = dqn_model.step()  
            total_reward += reward
            state = next_state
        list_reward.append(total_reward)
        losses.append(sum(dqn_model.list_loss)/len(dqn_model.list_loss))
        dqn_model.total_score = dqn_model.deep_env.get_final_score()
        scores_finaux.append(dqn_model.total_score)  
        steps.append(dqn_model.nb_step)  


    end_time = time.time()  
    total_time = end_time - start_time

    list_reward = [a/b for a, b in zip(list_reward, steps)]

    if total_time > 3600: 
        hours = total_time // 3600
        minutes = (total_time % 3600) // 60
        seconds = (total_time % 3600) % 60
        print(f"Temps d'exécution total : {hours} heures, {minutes} minutes et {seconds} secondes")
    elif total_time > 60: 
        minutes = total_time // 60
        seconds = total_time % 60
        print(f"Temps d'exécution total : {minutes} minutes et {seconds} secondes")
    else:  
        print(f"Temps d'exécution total : {total_time} secondes")

    print(f"Score moyen sur {episodes} jeux: {sum(scores_finaux) / episodes}")
    print(f"Loss moyenne sur {episodes} jeux: {sum(losses) / episodes}")
    print(f"Reward moyen sur {episodes} jeux: {sum(list_reward) / episodes}")


    data = pd.DataFrame({
        'Épisodes': range(1, len(scores_finaux) + 1),
        'Score moyen': scores_finaux,
        'Loss par épisode': losses,
        'Reward par épisode': list_reward
    })

    # Calculer la moyenne mobile pour lisser les courbes
    data['Score moyen MA'] = data['Score moyen'].rolling(window=30).mean()
    data['Loss par épisode MA'] = data['Loss par épisode'].rolling(window=30).mean()
    data['Reward par épisode MA'] = data['Reward par épisode'].rolling(window=30).mean()

    episode_list = np.arange(1, episodes + 1)

    # Score moyen par épisode
    plt.figure(figsize=(10, 6))
    plt.plot(episode_list, data['Score moyen MA'], label='Score moyen', color='blue')
    plt.title('Score Moyen par Épisode')
    plt.xlabel('Épisodes')
    plt.ylabel('Score moyen')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Perte moyenne par épisode
    plt.figure(figsize=(10, 6))
    plt.plot(episode_list, data['Loss par épisode MA'], label='Perte (Loss) par épisode', color='red')
    plt.title('Perte Moyenne par Épisode')
    plt.xlabel('Épisodes')
    plt.ylabel('Perte moyenne')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Récompense moyenne par épisode
    plt.figure(figsize=(10, 6))
    plt.plot(episode_list, data['Reward par épisode MA'], label='Récompense par épisode', color='green')
    plt.title('Récompense Moyenne par Épisode')
    plt.xlabel('Épisodes')
    plt.ylabel('Récompense moyenne')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()