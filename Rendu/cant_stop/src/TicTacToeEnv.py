import numpy as np
import random

class TicTacToeEnv:
    """
    Une classe pour représenter un environnement de jeu de TicTacToe adapté pour l'apprentissage en profondeur.
    """
    OBS_SIZE = 9  # La grille de TicTacToe est 3x3
    ACTION_SIZE = 9  # Nombre d'actions possibles (une pour chaque case de la grille)

    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 pour le joueur X, -1 pour le joueur O
        self.done = False
        self.winner = None
        self.reset()

    def reset(self):
        """
        Réinitialise l'état de l'environnement pour un nouveau jeu.
        """
        self.board.fill(0)
        self.current_player = 1 if random.choice([True, False]) else -1  # Commencer aléatoirement
        self.done = False
        self.winner = None
        return self.get_obs()

    def step(self, action):
        """
        Exécute une action dans l'environnement et met à jour son état.
        :param action: L'index de l'action choisie (0-8).
        :return: Un tuple contenant la nouvelle observation, le score, un booléen indiquant si le jeu est terminé, et informations supplémentaires.
        """
        if self.done:
            return self.get_obs(), 0, True, {}

        x, y = divmod(action, 3)
        if self.board[x, y] == 0:  # Vérifier si l'action est valide
            self.board[x, y] = self.current_player
            if self.check_winner(self.current_player):
                self.done = True
                self.winner = self.current_player
                reward = 1
            elif np.all(self.board != 0):
                self.done = True
                reward = 0.5  # Match nul
            else:
                reward = 0
            self.current_player *= -1  # Changer de joueur
        else:
            reward = -1  # Pénalité pour un mouvement invalide
            self.done = True  # Terminer le jeu sur un mouvement invalide

        return self.get_obs(), reward, self.done, {}

    def get_obs(self):
        """
        Retourne l'observation de l'état actuel du jeu pour le joueur actuel.
        """
        return np.copy(self.board.flatten()) * self.current_player  # Vue du joueur actuel

    def check_winner(self, player):
        """
        Vérifie si le joueur spécifié a gagné.
        """
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def render(self):
        """
        Affiche l'état actuel du jeu dans la console.
        """
        symbols = {1: 'X', -1: 'O', 0: ' '}
        for row in self.board:
            print(' '.join(symbols[x] for x in row))
        print()
