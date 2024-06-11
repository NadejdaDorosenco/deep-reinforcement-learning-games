from GridWorldEnv import GridWorldEnv
from LineWorldEnv import LineWorldEnv
from TicTacToeEnv import TicTacToeEnv
from CTEnv import GameEnv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

import time

class QNetwork(nn.Module):
    """
    Implémente un réseau de neurones pour l'apprentissage Q.

    Args:
        state_size (int): Taille de l'espace d'état.
        action_size (int): Taille de l'espace d'action.
        hidden_size (int): Taille de la couche cachée du réseau (par défaut: 64).
    """

    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        """
        Passe avant du réseau de neurones.

        Args:
            state (torch.Tensor): L'état actuel.

        Returns:
            torch.Tensor: Les valeurs d'action prédites pour l'état donné.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayMemory:
    """
    Implémente une mémoire de relecture pour stocker et échantillonner des expériences.

    Args:
        capacity (int): Capacité maximale de la mémoire de relecture.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        """
        Ajoute une transition à la mémoire de relecture.

        Args:
            transition (tuple): Transition à ajouter à la mémoire.
        """
        self.memory.append(transition)

    def sample(self, batch_size):
        """
        Échantillonne un lot d'expériences de la mémoire de relecture.

        Args:
            batch_size (int): Taille du lot à échantillonner.

        Returns:
            list: Lot d'expériences échantillonnées.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Retourne la taille actuelle de la mémoire de relecture.

        Returns:
            int: Taille de la mémoire de relecture.
        """
        return len(self.memory)

class DQNAgent:
    """
    Implémente un agent DQN (Deep Q-Network) pour l'apprentissage par renforcement.

    Args:
        state_size (int): Taille de l'espace d'état.
        action_size (int): Taille de l'espace d'action.
        hidden_size (int): Taille de la couche cachée du réseau (par défaut: 64).
        batch_size (int): Taille du lot pour l'entraînement (par défaut: 64).
        gamma (float): Facteur de remise (par défaut: 0.99).
        lr (float): Taux d'apprentissage (par défaut: 5e-4).
        memory_size (int): Capacité de la mémoire de relecture (par défaut: 10000).
        update_every (int): Fréquence de mise à jour du réseau cible (par défaut: 5).
    """

    def __init__(self, state_size, action_size, hidden_size=64, batch_size=64, gamma=0.99, lr=5e-4, memory_size=10000, update_every=5):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_every = update_every
        self.t_step = 0

        self.qnetwork_local = QNetwork(state_size, action_size, hidden_size)
        self.qnetwork_target = QNetwork(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_size)

    def step(self, state, action, reward, next_state, done):
        """
        Effectue une étape d'apprentissage à partir de l'expérience donnée.

        Args:
            state (tuple): L'état actuel.
            action (int): L'action prise dans cet état.
            reward (float): La récompense reçue après avoir pris l'action.
            next_state (tuple): L'état suivant après avoir pris l'action.
            done (bool): Indique si l'épisode est terminé après cette action.

        Returns:
            float or None: La perte d'entraînement si la mise à jour du réseau est effectuée, sinon None.
        """
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                return self.learn(experiences)
            return None

    def act(self, state, eps=0.01):
        """
        Sélectionne une action à prendre en fonction de l'état donné.

        Args:
            state (tuple or np.ndarray): L'état actuel.
            eps (float): Paramètre d'exploration epsilon (par défaut: 0.01).

        Returns:
            int: L'action sélectionnée.
        """
        if not isinstance(state, np.ndarray):
            state = np.array([state], dtype=np.float32)

        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # S'assurer qu'il y a des actions disponibles
        if self.action_size > 0:
            if random.random() > eps:
                return np.argmax(action_values.cpu().data.numpy())
            else:
                return random.choice(np.arange(self.action_size))
        else:
            # Retourner une valeur par défaut ou gérer l'absence d'action valide
            print("Aucune action valide disponible.")
            return None

    def learn(self, experiences):
        """
        Effectue une étape d'apprentissage en utilisant les expériences données.

        Args:
            experiences (list): Liste d'expériences à utiliser pour l'apprentissage.

        Returns:
            float: La perte d'entraînement pour cette étape.
        """
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        return loss.item()

    def soft_update(self, local_model, target_model, tau=1e-3):
        """
        Met à jour le modèle cible (target_model) en utilisant l'interpolation douce.

        Args:
            local_model (nn.Module): Le modèle local à partir duquel copier les paramètres.
            target_model (nn.Module): Le modèle cible à mettre à jour.
            tau (float): Paramètre de mélange (par défaut: 1e-3).
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

if __name__ == "__main__":
    env = GameEnv(num_players=2, player_colors=['Red', 'Blue'])
    dummy_state = env.reset()  # Reset pourrait ne pas mettre à jour action_vector si cela dépend des dés
    env.roll_dice_and_generate_actions()  
    agent = DQNAgent(state_size=GameEnv.OBS_SIZE, action_size=len(env.action_vector))
    
   # Initialisation de l'env et de l'agent DDQN pour l'environnement TicTacToeEnv
    #env = TicTacToeEnv()
    #dummy_state = env.reset()
    #agent = DQNAgent(state_size=TicTacToeEnv.OBS_SIZE, action_size=TicTacToeEnv.ACTION_SIZE)
    
    # Initialisation de l'env de l'agent DDQN pour l'environnement GridWorld
    #env = GridWorld()
    #dummy_state = env.reset()
    #agent = DQNAgent(state_size=GridWorld.OBS_SIZE, action_size=GridWorld.ACTION_SIZE)
    
    # Initialisation de l'env de l'agentt DDQN pour l'environnement LineWorld
    #env = LineWorld()
    #dummy_state = env.reset()
    #agent = DQNAgent(state_size=1, action_size=2)
    
    num_episodes = 100000
    max_t = 1000
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    total_score = 0  
    scores = []
    steps_per_episode = []  # Pour stocker le nombre de pas par épisode
    losses = []  # Pour stocker les pertes
    
    start_time = time.time()

    for i_episode in range(1, num_episodes+1):
        state = env.reset()
        score = 0
        steps = 0  # Compteur de pas pour cet épisode
        episode_losses = []  # Stocker les pertes pour cet épisode
        for t in range(max_t):
            action = agent.act(state, eps=eps_start)
            next_state, reward, done, _ = env.step(action)
            loss = agent.step(state, action, reward, next_state, done)
            if loss:  # Si une perte a été retournée
                episode_losses.append(loss)
            state = next_state
            score += reward
            steps += 1
            if done:
                break
        total_score += score
        scores.append(score)
        steps_per_episode.append(steps)  # Ajouter le nombre de pas pour cet épisode
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        losses.append(avg_loss)  # Ajouter la perte moyenne pour cet épisode
        eps_start = max(eps_end, eps_decay*eps_start)  # decrease epsilon

        #print(f"Episode {i_episode}\tScore: {score}")
    
    end_time = time.time()
    
    print("Algo DQN")
    # Calculer et afficher le score moyen sur tous les épisodes
    average_score = total_score / num_episodes
    print(f"Score moyen sur {num_episodes} épisodes: {average_score:.2f}")
    total_time = end_time - start_time
    print(f"Temps d'exécution total pour {num_episodes} épisodes: {total_time:.2f} secondes")

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)
    plt.plot(scores, label='Scores par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Score')
    plt.title('Progression des scores au cours de l\'entraînement DQN')
    plt.text(0, average_score + max(scores) * 0.01, f'{average_score:.2f}', color='r')
    plt.axhline(y=average_score, color='r', linestyle='-', label=f'Score moyen: {average_score:.2f}')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.subplot(2, 1, 2)
    plt.plot(losses, label='Loss par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Loss')
    plt.title('Loss par épisode DQN')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
