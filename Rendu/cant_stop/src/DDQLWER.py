from CTEnv import GameEnv
from GridWorldEnv import GridWorldEnv
from LineWorldEnv import LineWorldEnv
from TicTacToeEnv import TicTacToeEnv
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
    """Réseau neuronal pour l'estimation des valeurs Q."""
    def __init__(self, state_size, action_size, hidden_size=128):
        """
        Initialisation du réseau neuronal.

        Args:
            state_size (int): Taille de l'espace d'état.
            action_size (int): Taille de l'espace d'action.
            hidden_size (int): Taille de la couche cachée. Par défaut, 128.
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        """
        Passe avant du réseau neuronal.

        Args:
            state (torch.Tensor): État d'entrée.

        Returns:
            torch.Tensor: Valeurs d'action Q.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayMemory:
    """Tampon de relecture pour stocker l'expérience passée."""
    def __init__(self, capacity):
        """
        Initialisation du tampon de relecture.

        Args:
            capacity (int): Capacité maximale du tampon.
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        """
        Ajoute une transition au tampon.

        Args:
            transition (tuple): Transition à ajouter.
        """
        self.memory.append(transition)

    def sample(self, batch_size):
        """
        Échantillonne des transitions du tampon.

        Args:
            batch_size (int): Taille de l'échantillon.

        Returns:
            list: Liste d'échantillons de transitions.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Retourne la taille actuelle du tampon.

        Returns:
            int: Taille du tampon.
        """
        return len(self.memory)

class DDQLWERAgent:
    """Agent utilisant la méthode Double DQN avec Experience Replay."""
    def __init__(self, state_size, action_size, hidden_size=64, batch_size=128, gamma=1, lr=1e-4, memory_size=50000, update_every=10):
        """
        Initialisation de l'agent.

        Args:
            state_size (int): Taille de l'espace d'état.
            action_size (int): Taille de l'espace d'action.
            hidden_size (int): Taille de la couche cachée. Par défaut, 64.
            batch_size (int): Taille du lot pour l'apprentissage. Par défaut, 128.
            gamma (float): Facteur de remise. Par défaut, 1.
            lr (float): Taux d'apprentissage pour l'optimiseur. Par défaut, 1e-4.
            memory_size (int): Taille de la mémoire de relecture. Par défaut, 50000.
            update_every (int): Fréquence de mise à jour du réseau cible. Par défaut, 10.
        """
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
        Prend une étape d'apprentissage.

        Args:
            state (array_like): État actuel.
            action (int): Action choisie.
            reward (float): Récompense reçue.
            next_state (array_like): Prochain état.
            done (bool): Indique si l'épisode est terminé.

        Returns:
            float or None: Perte si l'apprentissage est effectué, sinon None.
        """
        self.memory.push((state, action, reward, next_state, done))
        
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                return self.learn(experiences)  # Retourne la perte ici
        return None  # Retourne None si pas d'apprentissage

    def act(self, state, eps=0.01):
        """
        Choix d'une action basée sur l'état actuel.

        Args:
            state (array_like): État actuel.
            eps (float): Probabilité d'exploration. Par défaut, 0.01.

        Returns:
            int: Action choisie.
        """
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0) if not isinstance(state, torch.Tensor) else state
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """
        Apprentissage à partir des expériences passées.

        Args:
            experiences (list): Liste d'expériences passées.

        Returns:
            float: Perte d'apprentissage.
        """
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Get max predicted Q values (for next states) from target model
        Q_locals_next = self.qnetwork_local(next_states).detach()
        _, best_actions = Q_locals_next.max(1)
        best_actions = best_actions.unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, best_actions)

        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        
        return loss.item()

    def soft_update(self, local_model, target_model, tau=1e-3):
        """
        Effectue une mise à jour douce des poids du modèle cible à partir du modèle local.
        
        Args:
            local_model (QNetwork): Le modèle Q à partir duquel mettre à jour les poids.
            target_model (QNetwork): Le modèle Q cible à mettre à jour.
            tau (float, optional): Le facteur de mise à jour douce. Default à 1e-3.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


if __name__ == "__main__":
    env = GameEnv(num_players=2, player_colors=['Red', 'Blue'])
    dummy_state = env.reset()  # Reset pourrait ne pas mettre à jour action_vector si cela dépend des dés
    env.roll_dice_and_generate_actions()  # Assurez-vous que cette méthode met à jour env.action_vector
    agent = DDQLWERAgent(state_size=GameEnv.OBS_SIZE, action_size=len(env.action_vector))
    
    # Initialisation de l'env et de l'agent DDQN pour l'environnement TicTacToeEnv
    #env = TicTacToeEnv()
    #dummy_state = env.reset()
    #agent = DDQLWERAgent(state_size=TicTacToeEnv.OBS_SIZE, action_size=TicTacToeEnv.ACTION_SIZE)
    
    # Initialisation de l'env de l'agent DDQN pour l'environnement GridWorld
    #env = GridWorld()
    #dummy_state = env.reset()
    #agent = DDQLWERAgent(state_size=GridWorld.OBS_SIZE, action_size=GridWorld.ACTION_SIZE)
    
    # Initialisation de l'env de l'agentt DDQN pour l'environnement LineWorld
    #env = LineWorld()
    #dummy_state = env.reset()
    #agent = DDQLWERAgent(state_size=1, action_size=2)

    num_episodes = 100000
    max_t = 1000
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 1
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

    print("Algo Double DQN With Experienced Replay")
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
    plt.title('Progression des scores au cours de l\'entraînement DDQNWER')
    plt.text(0, average_score + max(scores) * 0.01, f'{average_score:.2f}', color='r')
    plt.axhline(y=average_score, color='r', linestyle='-', label=f'Score moyen: {average_score:.2f}')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.subplot(2, 1, 2)
    plt.plot(losses, label='Loss par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Loss')
    plt.title('Loss par épisode DDQNWER')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    plt.show()
