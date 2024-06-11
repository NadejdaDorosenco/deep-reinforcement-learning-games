from DeepEnv import DeepEnv
from model import BalloonGameSolo
import numpy as np
import random

class RandomRollout:
    def __init__(self, env, num_games=1):
        self.env = env
        self.num_games = num_games
        print(f"Initialisation pour simuler {self.num_games} jeux.")
        
    def step(self):
        print("\nDébut d'un nouveau tour.")
        self.env.roll_dice()  # Assurez-vous que cela est nécessaire selon votre logique de jeu.
        state_vector = self.env.get_game_state_vector()
        done = False
        reward_total = 0

        while not done:
            num_dice = self.env.get_rolls()
            action_mask = self.env.available_actions_mask()
            valid_actions = [i for i, valid in enumerate(action_mask) if valid]
            action = np.random.choice(valid_actions)
            print(f"Action choisie: {action}")

            if action > 0:
                reroll_indices = self.env.get_reroll_indices(action)
                self.env.roll_dice(reroll_indices)
            else:
                # Si l'action 0 est choisie, cela peut signifier ne pas relancer. Assurez-vous que votre logique de jeu gère cela correctement.
                print("Aucun dés relancé.")

            reward, next_state, game_over = self.env.step()
            reward_total += reward
            done = game_over or self.env.get_game_over()  # Assurez-vous que cette vérification est correctement implémentée.

            print(f"Récompense: {reward}, Total des récompenses: {reward_total}, Terminé: {done}")

        return next_state, reward_total, done




    def play_turn(self):
        print("\nRéinitialisation de l'environnement pour un nouveau jeu.")
        self.env.reset()
        _, reward, done = self.step()
        final_score = self.env.get_final_score() if done else 0
        print(f"Score final du jeu: {final_score}")
        return final_score

    def simulate_multiple_games(self):
        print(f"Simulation de {self.num_games} jeux...")
        scores = [self.play_turn() for _ in range(self.num_games)]
        mean_score = np.mean(scores)
        return mean_score, scores

# Initialisation de l'environnement et du modèle
env = DeepEnv(BalloonGameSolo())
random_rollout_model = RandomRollout(env, num_games=1)  # Réduit à 10 pour le débogage

# Simulation des jeux et calcul de la moyenne des scores
mean_score, scores = random_rollout_model.simulate_multiple_games()
print(f"\nMoyenne des scores sur {random_rollout_model.num_games} jeux : {mean_score}")
