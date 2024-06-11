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
    """
    Un réseau de neurones simple pour approximer la fonction Q.
    
    Attributs :
        fc1 (torch.nn.Linear): Première couche entièrement connectée.
        fc2 (torch.nn.Linear): Deuxième couche entièrement connectée.
        fc3 (torch.nn.Linear): Troisième couche entièrement connectée, produisant l'estimation de Q pour chaque action.
    
    Args:
        state_size (int): Taille de l'état d'entrée.
        action_size (int): Nombre d'actions possibles.
        hidden_size (int): Nombre de neurones dans les couches cachées.
    """
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        """
        Propage en avant l'état d'entrée à travers le réseau pour obtenir les valeurs Q.
        
        Args:
            state (torch.Tensor): Le tenseur représentant l'état d'entrée.
            
        Returns:
            torch.Tensor: Les valeurs Q estimées pour chaque action.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SimpleDDQNAgent:
    """
    Implémente un agent utilisant Double Deep Q-Learning (DDQN).
    
    Args:
        state_size (int): La taille de l'état d'entrée.
        action_size (int): Le nombre d'actions possibles.
        hidden_size (int, optional): La taille des couches cachées du réseau Q. Default à 64.
        gamma (float, optional): Le facteur de remise. Default à 0.99.
        lr (float, optional): Le taux d'apprentissage. Default à 5e-4.
    """
    def __init__(self, state_size, action_size, hidden_size=64, gamma=0.99, lr=5e-4):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        self.qnetwork_local = QNetwork(state_size, action_size, hidden_size)
        self.qnetwork_target = QNetwork(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

    def step(self, state, action, reward, next_state, done):
        """
        Traite une étape d'expérience (state, action, reward, next_state, done) et met à jour le modèle.
        
        Args:
            state: L'état actuel.
            action: L'action effectuée.
            reward: La récompense reçue.
            next_state: L'état suivant atteint.
            done: Un booléen indiquant si l'épisode est terminé.
        
        Returns:
            float: La valeur de la perte calculée lors de la mise à jour.
        """
        # Convertir state et next_state en np.ndarray s'ils sont sous forme de tuple
        if isinstance(state, tuple):
            state = np.array(state)
        if isinstance(next_state, tuple):
            next_state = np.array(next_state)

        # Conversion des états pour PyTorch
        state = torch.from_numpy(state).float().unsqueeze(0)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)
        action = torch.tensor([[action]], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float)
        done = torch.tensor([done], dtype=torch.float)

        return self.learn(state, action, reward, next_state, done)

    def act(self, state, eps=0.01):
        """
        Sélectionne une action pour l'état donné en suivant une politique epsilon-greedy.
        
        Args:
            state: L'état actuel.
            eps (float, optional): Le paramètre epsilon pour la politique epsilon-greedy. Default à 0.01.
        
        Returns:
            int: L'action choisie.
        """
        if not isinstance(state, np.ndarray):
            state = np.array([state], dtype=np.float32)
            
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Sélection epsilon-greedy
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, state, action, reward, next_state, done):
        """
        Met à jour les poids du réseau à partir d'un batch d'expériences.
        
        Args:
            state: L'état de départ.
            action: L'action effectuée.
            reward: La récompense reçue.
            next_state: L'état suivant.
            done: Indique si l'épisode est terminé.
        
        Returns:
            float: La valeur de la perte après la mise à jour.
        """
        Q_targets_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)
        Q_targets = reward + (self.gamma * Q_targets_next * (1 - done))
        Q_expected = self.qnetwork_local(state).gather(1, action)

        # Calculer et appliquer la perte
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Mettre à jour le réseau cible
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
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

if __name__ == "__main__":
    env = GameEnv(num_players=2, player_colors=['Red', 'Blue'])
    dummy_state = env.reset()  # Reset pourrait ne pas mettre à jour action_vector si cela dépend des dés
    env.roll_dice_and_generate_actions()  # Assurez-vous que cette méthode met à jour env.action_vector
    agent = SimpleDDQNAgent(state_size=GameEnv.OBS_SIZE, action_size=len(env.action_vector))
    
    # Initialisation de l'env et de l'agent DDQN pour l'environnement TicTacToeEnv
    #env = TicTacToeEnv()
    #dummy_state = env.reset()
    #agent = SimpleDDQNAgent(state_size=TicTacToeEnv.OBS_SIZE, action_size=TicTacToeEnv.ACTION_SIZE)
    
    # Initialisation de l'env de l'agent DDQN pour l'environnement GridWorld
    #env = GridWorld()
    #dummy_state = env.reset()
    #agent = SimpleDDQNAgent(state_size=GridWorld.OBS_SIZE, action_size=GridWorld.ACTION_SIZE)
    
    # Initialisation de l'env de l'agentt DDQN pour l'environnement LineWorld
    #env = LineWorld()
    #dummy_state = env.reset()
    #agent = SimpleDDQNAgent(state_size=1, action_size=2)

    num_episodes = 10000
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
    
    print("Algo Double DQN")
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
    plt.title('Progression des scores au cours de l\'entraînement DDQN')
    plt.text(0, average_score + max(scores) * 0.01, f'{average_score:.2f}', color='r')
    plt.axhline(y=average_score, color='r', linestyle='-', label=f'Score moyen: {average_score:.2f}')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.subplot(2, 1, 2)
    plt.plot(losses, label='Loss par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Loss')
    plt.title('Loss par épisode DDQN')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    plt.show()
