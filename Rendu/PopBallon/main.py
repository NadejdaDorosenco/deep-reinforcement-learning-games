from DeepEnv import DeepEnv
from controller import GameController
from model import BalloonGameSolo
from view import PygameView
import deepQNetwork
import doubleDeepQNetwork
import time
import matplotlib.pyplot as plt

def start_game():
    choice = input("Voulez-vous jouer en mode solo avec l'interface graphique ou voulez-vous une simulation de jeu avec joueur random? Entrez solo, deepq, doubleDeepQ ou random :")

    if choice.lower() == 'solo':
        model = BalloonGameSolo()
        view = PygameView(model)
        controller = GameController(model, view)
        controller.run()
    elif choice.lower() == 'random':
        evaluation_points = [1000, 10000, 100000]  # Points d'évaluation souhaités
        for num_simulations in evaluation_points:
            total_scores = []
            total_rounds = []

            start_time = time.time()
            for _ in range(num_simulations):
                game = BalloonGameSolo()
                total_score, rounds = game.simulate_game()
                total_scores.append(total_score)
                total_rounds.append(rounds)

            elapsed_time = time.time() - start_time
            average_score = sum(total_scores) / num_simulations
            average_rounds = sum(total_rounds) / num_simulations

            print(f"Après {num_simulations} simulations (Point d'évaluation):")
            print(f"- Temps écoulé: {elapsed_time:.2f} secondes ({num_simulations / elapsed_time:.2f} parties/seconde)")
            print(f"- Score moyen: {average_score:.2f}")
            print(f"- Tours moyens par partie: {average_rounds:.2f}\n")
    elif choice.lower() == 'deepq':
        deepQNetwork.main() 
    elif choice.lower() == 'doubledeepq':
        doubleDeepQNetwork.main()

    else:
        print("Choix invalide. Veuillez taper 'solo', 'deepQ', 'deepqrelay', 'doubledeepQ', 'DDQprioritizedReplay' ou 'random'.")

if __name__ == "__main__":
    start_game()
