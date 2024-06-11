import random
import matplotlib.pyplot as plt
from CTEnv import GameEnv

import random

class RandomRollout:
    def __init__(self, env):
        self.env = env

    def play_game(self):
        """
        Joue un jeu complet en choisissant des actions aléatoirement jusqu'à ce que le jeu soit terminé.
        """
        state = self.env.reset()
        game_over = False
        total_reward = 0

        while not game_over:
            # Assumer que les actions valides sont représentées par des indices dans action_vector
            # et que la première action (index 0) est toujours "arrêter le tour",
            # ce qui n'est pas toujours une action valide au début d'un tour.
            # Générer l'action_vector avec les combinaisons de dés possibles pour le tour actuel
            dice_results = self.env.roll_dice_and_generate_actions()  # Cette méthode doit être implémentée dans GameEnv
            action_size = len(self.env.action_vector)

            # Choisir une action valide aléatoirement (excepté l'arrêt si pas permis)
            action_index = random.choice(range(1, action_size)) if action_size > 1 else 0

            state, reward, game_over, _ = self.env.step(action_index)
            total_reward += reward

        print(f"Jeu terminé. Récompense totale: {total_reward}")
        return total_reward

    def evaluate_strategy(self, num_games):
        """
        Évalue la stratégie de rollout aléatoire sur un certain nombre de jeux.
        """
        total_rewards = 0
        for _ in range(num_games):
            self.play_game()
            total_rewards += self.play_game()

        average_reward = total_rewards / num_games
        print(f"Récompense moyenne sur {num_games} jeux: {average_reward}")

if __name__ == "__main__":
    env = GameEnv(num_players=2, player_colors=['Red', 'Blue'])
    random_rollout = RandomRollout(env)

    # Pour jouer un seul jeu
    #random_rollout.play_game()

    ## Pour évaluer la stratégie sur plusieurs jeux
    scores, average_reward = random_rollout.evaluate_strategy(num_games=100000)
    plt.figure(figsize=(10,6))
    plt.plot(scores, label='Scores par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Score')
    plt.title('Progression des scores au cours de l\'entraînement')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.text(-0.05*len(scores), max(scores)*1.05, f'Reward moyen: {average_reward:.2f}', ha='left', va='top')
    plt.tight_layout() 
    plt.show()
