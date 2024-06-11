import numpy as np
from CTEnv import GameEnv
from GridWorldEnv import GridWorldEnv
from LineWorldEnv import LineWorldEnv
from TicTacToeEnv import TicTacToeEnv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt
import time

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        """
        Initialise un réseau de neurones pour l'algorithme de l'apprentissage par renforcement profond.
        
        :param state_size: Taille de l'espace des états.
        :param action_size: Taille de l'espace des actions.
        :param hidden_size: Taille de la couche cachée du réseau de neurones.
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        """
        Passe avant du réseau de neurones.
        
        :param state: L'état en entrée.
        :return: Les valeurs d'action prédites pour l'état donné.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        """
        Initialise une mémoire de relecture prioritaire.
        
        :param capacity: Capacité maximale de la mémoire.
        :param alpha: Paramètre de priorité, déterminant le degré de priorité.
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, error, transition):
        """
        Ajoute une transition à la mémoire avec une priorité donnée.
        
        :param error: Erreur associée à la transition, utilisée pour calculer la priorité.
        :param transition: Transition à ajouter à la mémoire.
        """
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio ** self.alpha
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Échantillonne un lot de transitions de la mémoire de relecture prioritaire.
        
        :param batch_size: Taille du lot à échantillonner.
        :param beta: Paramètre beta pour ajuster l'importance des priorités lors de l'échantillonnage.
        :return: Un lot d'échantillons, les indices correspondants et les poids pour chaque échantillon.
        """
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios / prios.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, errors):
        """
        Met à jour les priorités des transitions échantillonnées.
        
        :param indices: Indices des transitions dans la mémoire.
        :param errors: Erreurs associées aux transitions.
        """
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error ** self.alpha

    def __len__(self):
        """
        Retourne la taille actuelle de la mémoire.
        """
        return len(self.buffer)

class DDQNWPERAgent:
    def __init__(self, state_size, action_size, hidden_size=64, batch_size=64, gamma=0.99, lr=1e-4, memory_size=50000, alpha=0.6, beta_start=0.4, beta_frames=10000):
        """
        Initialise un agent basé sur l'algorithme Double DQN avec expérience de relecture prioritaire.
        
        :param state_size: Taille de l'espace des états.
        :param action_size: Taille de l'espace des actions.
        :param hidden_size: Taille de la couche cachée du réseau de neurones.
        :param batch_size: Taille des lots pour l'apprentissage par mini-lots.
        :param gamma: Facteur de remise pour les récompenses futures.
        :param lr: Taux d'apprentissage pour l'optimiseur.
        :param memory_size: Taille de la mémoire de relecture.
        :param alpha: Paramètre de priorité pour la mémoire de relecture prioritaire.
        :param beta_start: Valeur initiale du paramètre beta pour l'échantillonnage basé sur la priorité.
        :param beta_frames: Nombre de trames pour l'annulation linéaire du paramètre beta.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0

        self.qnetwork_local = QNetwork(state_size, action_size, hidden_size)
        self.qnetwork_target = QNetwork(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.memory = PrioritizedReplayMemory(memory_size, alpha)
        
    def act(self, state, eps=0.0):
        """
        Retourne une action pour un état donné en suivant une stratégie epsilon-greedy.
        
        :param state: L'état actuel de l'environnement.
        :param eps: Le paramètre epsilon pour la stratégie epsilon-greedy.
        :return: L'action choisie.
        """
        state = np.array(state) if isinstance(state, tuple) else state
        device = next(self.qnetwork_local.parameters()).device  # Obtenir le dispositif à partir des paramètres du modèle
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        """
        Effectue une étape d'apprentissage basée sur la transition donnée.
        
        :param state: L'état actuel.
        :param action: L'action choisie.
        :param reward: La récompense reçue pour l'action.
        :param next_state: L'état suivant après avoir pris l'action.
        :param done: Un indicateur indiquant si l'épisode est terminé ou non.
        :return: La perte de l'étape d'apprentissage.
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

    def beta_by_frame(self, frame):
        """
        Calcule la valeur du paramètre beta pour l'échantillonnage basé sur la priorité.
        
        :param frame: Le numéro de la trame.
        :return: La valeur de beta correspondante.
        """
        return min(1.0, self.beta_start + frame * (1.0 - self.beta_start) / self.beta_frames)

    def learn(self, state, action, reward, next_state, done):
        """
        Effectue une étape d'apprentissage basée sur la transition donnée.
        
        :param state: L'état actuel.
        :param action: L'action choisie.
        :param reward: La récompense reçue pour l'action.
        :param next_state: L'état suivant après avoir pris l'action.
        :param done: Un indicateur indiquant si l'épisode est terminé ou non.
        :return: La perte de l'étape d'apprentissage.
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
        Met à jour les paramètres du réseau cible de manière douce.
        
        :param local_model: Le modèle local utilisé pour mettre à jour les poids.
        :param target_model: Le modèle cible dont les poids sont mis à jour.
        :param tau: Le coefficient de lissage pour la mise à jour douce.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


if __name__ == "__main__":
    env = GameEnv(num_players=2, player_colors=['Red', 'Blue'])
    
    dummy_state = env.reset()  # Reset pourrait ne pas mettre à jour action_vector si cela dépend des dés
    env.roll_dice_and_generate_actions()  # Assurez-vous que cette méthode met à jour env.action_vector
    agent = DDQNWPERAgent(state_size=GameEnv.OBS_SIZE, action_size=len(env.action_vector))
    
    # Initialisation de l'env et de l'agent DDQN pour l'environnement TicTacToeEnv
    #env = TicTacToeEnv()
    #dummy_state = env.reset()
    #agent = DDQNWPERAgent(state_size=TicTacToeEnv.OBS_SIZE, action_size=TicTacToeEnv.ACTION_SIZE)
    
    # Initialisation de l'env de l'agent DDQN pour l'environnement GridWorld
    #env = GridWorld()
    #dummy_state = env.reset()
    #agent = DDQNWPERAgent(state_size=GridWorld.OBS_SIZE, action_size=GridWorld.ACTION_SIZE)
    
    # Initialisation de l'env de l'agentt DDQN pour l'environnement LineWorld
    #env = LineWorld()
    #dummy_state = env.reset()
    #agent = DDQNWPERAgent(state_size=1, action_size=2)

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
    
    print("Algo Double DQN With Prioritized Experienced Replay")
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
    plt.title('Progression des scores au cours de l\'entraînement DDQNWPER')
    plt.text(0, average_score + max(scores) * 0.01, f'{average_score:.2f}', color='r')
    plt.axhline(y=average_score, color='r', linestyle='-', label=f'Score moyen: {average_score:.2f}')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.subplot(2, 1, 2)
    plt.plot(losses, label='Loss par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Loss')
    plt.title('Loss par épisode DDQNWPER')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    plt.show()
