import math
import random
import numpy as np

from gameEnv import GameEnv

class MCTSNode:
    """Noeud de l'arbre de recherche MCTS pour le jeu."""
    def __init__(self, game_env, parent=None, parent_action=None):
        """
        Initialise un nœud de l'arbre MCTS.

        Args:
            game_env (GameEnv): L'environnement du jeu.
            parent (MCTSNode, optional): Le nœud parent. Par défaut, None.
            parent_action (int, optional): L'action effectuée par le parent pour atteindre ce nœud. Par défaut, None.
        """
        self.game_env = game_env
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.visits = 0
        self.value = 0
        self.untried_actions = self.untried_actions()
        self.player_index = game_env.current_player_index

    def untried_actions(self):
        """
        Génère des actions possibles pour cet état du jeu.

        Returns:
            list: Liste des indices des actions possibles.
        """
        actions = []
        for i, action in enumerate(self.game_env.action_vector):
            actions.append(i)
        return actions

    def expand(self):
        """
        Étend le nœud en créant un enfant.

        Returns:
            MCTSNode: Le nœud enfant étendu.
        """
        # Vérifiez d'abord si des actions non essayées restent
        if not self.untried_actions:
            return None  # Aucune action à étendre si vide

        action = self.untried_actions.pop(0)  # Utilisez le premier élément pour éviter les index hors de portée
        if action >= len(self.game_env.action_vector):
            print("Erreur: L'index d'action est hors de portée.")
            return None

        next_state = self.game_env.clone_stochastic()
        # Assurez-vous que next_state a son action_vector à jour avant d'appeler step
        next_state.roll_dice_and_generate_actions()
        next_state.step(action)  # Assurez-vous que cette opération est valide
        child_node = MCTSNode(next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        """
        Vérifie si le jeu est terminé dans cet état.

        Returns:
            bool: True si le jeu est terminé, False sinon.
        """
        return self.game_env.get_game_over()

    def rollout_policy(self, possible_moves):
        """
        Politique de simulation pour sélectionner une action au hasard parmi les actions possibles.

        Args:
            possible_moves (list): Liste des indices des actions possibles.

        Returns:
            int: L'indice de l'action sélectionnée.
        """
        # Sélectionne une action au hasard parmi les actions possibles
        return random.choice(possible_moves)

    def rollout(self):
        """
        Effectue une simulation de rollout à partir de cet état du jeu.

        Returns:
            int: Le score obtenu à la fin de la simulation.
        """
        current_rollout_state = self.game_env.clone_stochastic()
        while not current_rollout_state.get_game_over():
            current_rollout_state.roll_dice_and_generate_actions()  # Mise à jour des actions possibles
            possible_moves = range(len(current_rollout_state.action_vector))
            if possible_moves:  # Vérifie si la liste n'est pas vide
                action = self.rollout_policy(list(possible_moves))
                current_rollout_state.step(action)
            else:
                break  # Sort de la boucle si aucune action possible
        return current_rollout_state.get_score()

    def backpropagate(self, reward):
        """
        Met à jour les statistiques du nœud à la suite d'une simulation.

        Args:
            reward (int): Le récompense ou le score obtenu à la fin de la simulation.
        """
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def best_child(self, c_param=1.4):
        """
        Sélectionne le meilleur enfant en utilisant l'UCT (Upper Confidence Bound Applied to Trees).

        Args:
            c_param (float, optional): Le paramètre d'exploration UCT. Par défaut, 1.4.

        Returns:
            MCTSNode: Le meilleur enfant sélectionné.
        """
        if not self.children:
            return None
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def select_child(self):
        """
        Sélectionne un enfant à explorer.

        Returns:
            MCTSNode: Le nœud enfant sélectionné.
        """
        # Expande le nœud si possible
        if self.untried_actions:
            return self.expand()
        elif self.children:  # Vérifie si le nœud a des enfants avant de sélectionner
            return self.best_child()
        else:
            return None  # Retourne None si le nœud est terminal ou sans enfants

class MCTSAgent:
    """Agent MCTS pour prendre des décisions dans le jeu."""
    def __init__(self, number_of_simulations):
        """
        Initialise l'agent MCTS avec le nombre de simulations.

        Args:
            number_of_simulations (int): Le nombre de simulations à exécuter pour chaque décision.
        """
        self.number_of_simulations = number_of_simulations

    def select_action(self, game_env):
        """
        Sélectionne une action à partir de l'environnement de jeu donné.

        Args:
            game_env (GameEnv): L'environnement de jeu actuel.

        Returns:
            int: L'indice de l'action sélectionnée.
        
        Raises:
            ValueError: Si aucune action valide n'est disponible dans l'état actuel du jeu.
        """
        # Clone l'environnement de jeu pour ne pas perturber l'état du jeu original
        cloned_env = game_env.clone_stochastic()
        cloned_env.roll_dice_and_generate_actions()  # Mise à jour des actions possibles
        root = MCTSNode(cloned_env)

        for _ in range(self.number_of_simulations):
            node = root
            while node is not None and not node.is_terminal_node():
                node.game_env.roll_dice_and_generate_actions()
                next_node = node.select_child()
                if next_node is None:  # Si aucun enfant n'est sélectionné, arrête la simulation
                    break
                node = next_node
            if node is not None:  # Vérifie si une simulation valide a été effectuée
                reward = node.rollout()
                node.backpropagate(reward)

        if root.children:
            return max(root.children, key=lambda x: x.visits).parent_action
        else:
            if len(game_env.action_vector) > 0:
                return random.choice(range(len(game_env.action_vector)))
            else:
                raise ValueError("Aucune action valide disponible dans l'état actuel.")
            
def play_game(env, mcts_agent):
    """
    Joue une partie complète du jeu.

    Args:
        env (GameEnv): L'environnement du jeu.
        mcts_agent (MCTSAgent): L'agent MCTS pour prendre des décisions.

    Returns:
        int: Le score final de la partie.
    """
    env.reset()
    while not env.get_game_over():
        action_index = mcts_agent.select_action(env)
        env.step(action_index)
    return env.get_score()

        

if __name__ == "__main__":
    num_episodes = 10

    env = GameEnv(num_players=2, player_colors=['Red', 'Blue'])
    mcts_agent = MCTSAgent(number_of_simulations=100)

    scores = []
    for episode in range(num_episodes):
        score = play_game(env, mcts_agent)
        scores.append(score)
        print(f"Épisode {episode + 1}: Score = {score}")

    average_score = sum(scores) / num_episodes
    print(f"Score moyen sur {num_episodes} épisodes: {average_score}")
