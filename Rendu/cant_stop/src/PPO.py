import time
import numpy as np
from CTEnv import GameEnv
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

class Actor(nn.Module):
    """Implémente le réseau de l'acteur pour l'algorithme PPO."""
    def __init__(self, input_dim, output_dim):
        """
        Initialise le réseau de l'acteur.

        Args:
            input_dim (int): Dimension de l'entrée.
            output_dim (int): Dimension de la sortie.
        """
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        """
        Effectue une passe avant du réseau de l'acteur.

        Args:
            state (torch.Tensor): L'état en entrée.

        Returns:
            torch.Tensor: Les probabilités de chaque action.
        """
        return self.network(state)

class Critic(nn.Module):
    """Implémente le réseau du critique pour l'algorithme PPO."""
    def __init__(self, input_dim):
        """
        Initialise le réseau du critique.

        Args:
            input_dim (int): Dimension de l'entrée.
        """
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        """
        Effectue une passe avant du réseau du critique.

        Args:
            state (torch.Tensor): L'état en entrée.

        Returns:
            torch.Tensor: La valeur estimée de l'état.
        """
        return self.network(state)

def compute_returns(next_value, rewards, masks, gamma=0.99):
    """
    Calcule les rendements actualisés pour chaque étape.

    Args:
        next_value (float): La valeur de l'état suivant.
        rewards (list): Liste des récompenses.
        masks (list): Liste des masques (1 pour les états terminaux, 0 sinon).
        gamma (float, optional): Le facteur de remise. Par défaut, 0.99.

    Returns:
        list: Liste des rendements actualisés.
    """
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    """
    Calcule l'estimation avantage généralisée (GAE).

    Args:
        next_value (torch.Tensor): La valeur de l'état suivant.
        rewards (list): Liste des récompenses.
        masks (list): Liste des masques (1 pour les états terminaux, 0 sinon).
        values (torch.Tensor): Les valeurs estimées des états.
        gamma (float, optional): Le facteur de remise. Par défaut, 0.99.
        tau (float, optional): Le paramètre tau pour GAE. Par défaut, 0.95.

    Returns:
        list: Liste des estimations GAE.
    """
    # S'assurer que next_value est de la bonne forme. 
    # Si next_value est un scalaire ou un tenseur 1D, ajoutez une dimension pour correspondre à values.
    if next_value.dim() == 0:
        next_value = next_value.unsqueeze(0)  # Ajoute une dimension si scalaire
    elif next_value.dim() == 1:
        next_value = next_value.unsqueeze(1)  # Transforme [batch_size] en [batch_size, 1]

    # Assurez-vous maintenant que values est également un tenseur 2D [batch_size, 1].
    # Cette opération devrait maintenant fonctionner sans erreur.
    values = torch.cat([values, next_value], dim=0)

    # Initialiser le calcul de GAE
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * gae * masks[step]
        returns.insert(0, gae + values[step])
    return returns


def ppo_update(agent, states, actions, log_probs_old, returns, advantages, gamma, tau, clip_param, ppo_epochs, mini_batch_size):
    """
    Met à jour les réseaux de politique et de valeur avec l'algorithme PPO.

    Args:
        agent (PPOAgent): L'agent PPO.
        states (torch.Tensor): Les états observés.
        actions (torch.Tensor): Les actions prises.
        log_probs_old (torch.Tensor): Les logarithmes des probabilités des actions prises précédemment.
        returns (torch.Tensor): Les rendements actualisés.
        advantages (torch.Tensor): Les estimations d'avantages généralisés.
        gamma (float): Le facteur de remise.
        tau (float): Le paramètre tau pour GAE.
        clip_param (float): Le paramètre de clipping.
        ppo_epochs (int): Le nombre d'époques pour la mise à jour de PPO.
        mini_batch_size (int): La taille du mini-lot pour la mise à jour de PPO.

    Returns:
        tuple: La perte moyenne de l'acteur et du critique sur toutes les époques.
    """
    total_actor_loss = 0
    total_critic_loss = 0
    
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    old_log_probs = torch.FloatTensor(log_probs_old)
    returns = torch.FloatTensor(returns)
    advantages = torch.FloatTensor(advantages)

    for _ in range(ppo_epochs):
        # Calcul de la nouvelle distribution de probabilité des actions
        probs = agent.actor(states)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        
        # Calcul du ratio des probabilités pour l'action choisie
        ratios = torch.exp(new_log_probs - old_log_probs)
        
        # Calcul de l'objectif de la politique en utilisant le clipping
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * advantages
        actor_loss = -torch.mean(torch.min(surr1, surr2))
        
        # Mise à jour de l'actor
        agent.optimizer_actor.zero_grad()
        actor_loss.backward()
        agent.optimizer_actor.step()
        
        # Calcul de la perte du critic en utilisant la différence des carrés entre les retours et les prédictions de valeur
        values = agent.critic(states).squeeze()
        critic_loss = torch.mean((returns - values) ** 2)
        
        # Mise à jour du critic
        agent.optimizer_critic.zero_grad()
        critic_loss.backward()
        agent.optimizer_critic.step()
        
        total_actor_loss += actor_loss.item()
        total_critic_loss += critic_loss.item()

    # Retourner la moyenne des pertes sur toutes les époques
    average_actor_loss = total_actor_loss / ppo_epochs
    average_critic_loss = total_critic_loss / ppo_epochs
    return average_actor_loss, average_critic_loss


class PPOAgent:
    """Implémente l'agent utilisant l'algorithme PPO."""
    def __init__(self, state_size, action_size, lr_actor=1e-4, lr_critic=1e-3):
        """
        Initialise l'agent PPO avec les paramètres d'apprentissage.

        Args:
            state_size (int): Dimension de l'espace d'état.
            action_size (int): Dimension de l'espace d'action.
            lr_actor (float, optional): Taux d'apprentissage pour l'acteur. Par défaut, 1e-4.
            lr_critic (float, optional): Taux d'apprentissage pour le critique. Par défaut, 1e-3.
        """
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = 0.99  # Discount factor
        self.tau = 0.95  # GAE parameter
        self.clip_param = 0.2
        self.ppo_epochs = 4
        self.mini_batch_size = 64

    def select_action(self, state):
        """
        Sélectionne une action à prendre basée sur l'état donné.

        Args:
            state (np.ndarray): L'état actuel.

        Returns:
            tuple: L'action sélectionnée et le logarithme de la probabilité de cette action.
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state)
        probs = probs / probs.sum()
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def update(self, states, action, log_prob_old, rewards, dones, next_states):
        """
        Met à jour les réseaux de politique et de valeur.

        Args:
            states (np.ndarray): Les états observés.
            action (int): L'action prise.
            log_prob_old (float): Le logarithme de la probabilité de l'action prise précédemment.
            rewards (float): La récompense reçue.
            dones (bool): Indique si l'état est terminal ou non.
            next_states (np.ndarray): Les prochains états observés.

        Returns:
            tuple: La perte moyenne de l'acteur et du critique sur toutes les époques.
        """
        states = torch.FloatTensor([states])
        next_states = torch.FloatTensor([next_states])
        actions = torch.LongTensor([action])
        log_probs_old = torch.FloatTensor([log_prob_old])
        rewards = torch.FloatTensor([rewards])
        dones = torch.FloatTensor([dones])
        
        # Get predicted next state values from the critic
        next_state_values = self.critic(next_states).detach()
        # Calculate returns and advantages
        returns = compute_returns(next_state_values, rewards, 1 - dones, self.gamma)
        advantages = compute_gae(next_state_values, rewards, 1 - dones, self.critic(states).detach(), self.gamma, self.tau)
        
        # Convert lists to tensors
        returns = torch.cat(returns).detach()
        advantages = torch.cat(advantages).detach()
        
        # Update policy and value networks
        loss = ppo_update(self, states, actions, log_probs_old, returns, advantages, self.gamma, self.tau, self.clip_param, self.ppo_epochs, self.mini_batch_size)
    
        return loss

if __name__ == "__main__":
    env = GameEnv(num_players=2, player_colors=['Red', 'Blue'])
    state = env.reset()
    env.roll_dice_and_generate_actions()
    agent = PPOAgent(state_size=GameEnv.OBS_SIZE, action_size=len(env.action_vector))

    num_episodes = 10000
    max_t = 1000
    scores = []
    losses = []
    total_score = 0

    start_time = time.time()

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        env.roll_dice_and_generate_actions()
        score = 0

        for t in range(max_t):
            action, log_prob = agent.select_action(state)
            if action >= len(env.action_vector):
                continue
            next_state, reward, done, _ = env.step(action)
            loss = agent.update(state, action, log_prob, reward, done, next_state)
            losses.append(loss) 
            state = next_state
            score += reward
            if done:
                break

        total_score += score
        scores.append(score)
        print(f"Episode {i_episode}\tScore: {score}")

    end_time = time.time()
    
    print("Algo PPO A2C Style")
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
    plt.title('Progression des scores au cours de l\'entraînement PPO')
    plt.text(0, average_score + max(scores) * 0.01, f'{average_score:.2f}', color='r')
    plt.axhline(y=average_score, color='r', linestyle='-', label=f'Score moyen: {average_score:.2f}')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.subplot(2, 1, 2)
    plt.plot(losses, label='Loss par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Loss')
    plt.title('Loss par épisode PPO')
    plt.legend()
    
    plt.tight_layout()
    plt.show()