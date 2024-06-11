import numpy as np
from CTEnv import GameEnv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple

class PolicyNetwork(nn.Module):
    """Implémente le réseau de politique pour l'algorithme REINFORCE."""
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialise le réseau de politique.

        Args:
            input_size (int): Taille de l'entrée.
            hidden_size (int): Taille de la couche cachée.
            output_size (int): Taille de la sortie.
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Effectue une passe avant du réseau de politique.

        Args:
            x (torch.Tensor): L'entrée du réseau.

        Returns:
            torch.Tensor: Les probabilités des actions.
        """
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

class REINFORCE:
    """Implémente l'algorithme REINFORCE."""
    def __init__(self, policy_network, num_actions):
        """
        Initialise l'algorithme REINFORCE.

        Args:
            policy_network (PolicyNetwork): Le réseau de politique utilisé.
            num_actions (int): Le nombre d'actions disponibles.
        """
        self.policy_network = policy_network
        self.optimizer = optim.Adam(policy_network.parameters(), lr=1e-2)
        self.saved_log_probs = []
        self.rewards = []
        self.num_actions = num_actions

    def select_action(self, state, action_mask):
        """
        Sélectionne une action à prendre basée sur l'état donné et le masque d'actions.

        Args:
            state (np.ndarray): L'état actuel.
            action_mask (np.ndarray): Le masque des actions disponibles.

        Returns:
            int: L'action sélectionnée.
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        logits = self.policy_network(state)
        
        # Vérifier la dimension de logits
        print("Dimension de logits:", logits.shape)
        
        # Assurer que l'action_mask est un Tensor avec la bonne dimension
        action_mask = torch.tensor(action_mask, dtype=torch.float)
        
        # Répéter action_mask pour qu'elle ait la même taille que logits
        action_mask = action_mask.repeat(logits.size(0), 1)
        
        # Vérifier la dimension de action_mask
        print("Dimension de action_mask:", action_mask.shape)
        
        # Appliquer l'action_mask aux logits
        masked_logits = logits + (action_mask - 1) * 1e9
        
        # Sélectionner une action
        action = torch.multinomial(F.softmax(masked_logits, dim=-1), 1).item()
        self.saved_log_probs.append(F.log_softmax(masked_logits, dim=-1)[0, action])
        return action


    def finish_episode(self, gamma=0.99):
        """
        Termine l'épisode et met à jour les poids du réseau de politique.

        Args:
            gamma (float, optional): Le facteur d'escompte. Par défaut, 0.99.
        """
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.saved_log_probs = []
        self.rewards = []

def main():
    """
    Fonction principale pour exécuter l'algorithme REINFORCE sur l'environnement du jeu.
    """
    env = GameEnv(num_players=2, player_colors=['Red', 'Blue'])
    num_actions = len(env.action_vector)  # Ajusté pour correspondre à la taille du vecteur d'actions
    policy_net = PolicyNetwork(env.OBS_SIZE, 128, num_actions)
    reinforce_agent = REINFORCE(policy_net, num_actions)

    for episode in range(1000):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            env.roll_dice_and_generate_actions()
            if len(env.action_vector) == 0:
                print("No valid actions available, check the update of env.action_vector.")
                continue 
            action_mask = env.available_action_mask()
            print("action mask", action_mask)
            print("len(action_mask)",len(action_mask))
            action = reinforce_agent.select_action(np.array(state), action_mask)
            state, reward, done, _ = env.step(action)
            reinforce_agent.rewards.append(reward)
            total_reward += reward
        reinforce_agent.finish_episode()
        print(f"Episode {episode+1}: Total Reward: {total_reward}")

if __name__ == '__main__':
    main()
