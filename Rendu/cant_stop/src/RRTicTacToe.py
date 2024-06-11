import random
import time
import matplotlib.pyplot as plt
from TicTacToeEnv import TicTacToeEnv

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
            action = random.choice(range(self.env.ACTION_SIZE))  # Choix aléatoire d'une action
            state, reward, game_over, _ = self.env.step(action)
            total_reward += reward

        print(f"Jeu terminé. Récompense totale: {total_reward}")
        return total_reward

    def evaluate_strategy(self, num_games):
        """
        Évalue la stratégie de rollout aléatoire sur un certain nombre de jeux.
        """
        total_rewards = 0
        scores = []
        for i in range(num_games):
            reward = self.play_game()
            total_rewards += reward
            scores.append(reward)
        average_reward = total_rewards / num_games
        print(f"Récompense moyenne sur {num_games} jeux: {average_reward}")
        return scores, average_reward

if __name__ == "__main__":
    env = TicTacToeEnv()

    start_time = time.time()
    random_rollout = RandomRollout(env)
    end_time = time.time()

    ## Pour jouer un seul jeu
    # random_rollout.play_game()

    ## Pour évaluer la stratégie sur plusieurs jeux
    num_games = 100000
    scores, average_reward = random_rollout.evaluate_strategy(num_games=num_games)
    
    print("Algo Random Rollout")
    print(f"Score moyen sur {num_games} épisodes: {average_reward:.2f}")
    total_time = end_time - start_time
    print(f"Temps d'exécution total pour {num_games} épisodes: {total_time:.2f} secondes")
    
    plt.figure(figsize=(10,6))
    plt.plot(scores, label='Scores par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Score')
    plt.title('Progression des scores au cours de l\'entraînement')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.text(-0.05*len(scores), max(scores)*1.05, f'Reward moyen: {average_reward:.2f}', ha='left', va='top')
    plt.tight_layout() 
    plt.show()
