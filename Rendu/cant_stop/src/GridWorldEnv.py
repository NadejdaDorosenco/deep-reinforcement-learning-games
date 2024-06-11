import numpy as np
import random

class GridWorldEnv:
    """
    Classe représentant un environnement de type Grid World pour l'apprentissage par renforcement.
    """
    OBS_SIZE = 2  # La position de l'agent sur la grille: (x, y)
    ACTION_SIZE = 4  # Haut, Bas, Gauche, Droite

    def __init__(self, grid_size=(5, 5), start_position=(0, 0), goal_position=(4, 4)):
        """
        Initialise un nouvel environnement de type Grid World.

        :param grid_size: La taille de la grille (largeur, hauteur).
        :param start_position: La position initiale de l'agent sur la grille.
        :param goal_position: La position à atteindre pour gagner.
        """
        self.grid_size = grid_size
        self.start_position = start_position
        self.goal_position = goal_position
        self.agent_position = start_position

    def reset(self):
        """
        Réinitialise l'environnement à son état initial.
        """
        self.agent_position = self.start_position
        return self.get_obs()

    def get_obs(self):
        """
        Retourne l'observation actuelle de l'environnement (la position de l'agent).
        """
        return np.array(self.agent_position)

    def step(self, action):
        """
        Exécute une action et met à jour l'état de l'environnement.

        :param action: L'indice de l'action choisie.
        :return: Un tuple contenant la nouvelle observation, la récompense, un booléen indiquant si le but est atteint, et des infos supplémentaires.
        """
        # Définir les déplacements possibles
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # Haut, Bas, Gauche, Droite
        
        # Calculer la nouvelle position
        delta = moves.get(action)
        new_position = (max(0, min(self.grid_size[0]-1, self.agent_position[0] + delta[0])),
                        max(0, min(self.grid_size[1]-1, self.agent_position[1] + delta[1])))
        
        self.agent_position = new_position
        
        # Vérifier si l'objectif est atteint
        done = self.agent_position == self.goal_position
        reward = 1 if done else -0.01  # Récompense pour atteindre le but, pénalité légère sinon
        
        return self.get_obs(), reward, done, {}

    def render(self):
        """
        Affiche l'état actuel de la grille pour visualisation.
        """
        grid = np.zeros(self.grid_size)
        grid[self.goal_position] = 2  # Marquer l'objectif
        grid[self.agent_position] = 1  # Marquer la position de l'agent
        print(grid)
        
    def clone_stochastic(self):
        """Crée une copie profonde de cet environnement pour la simulation."""
        cloned_env = GridWorldEnv(self.grid_size, self.start_position, self.goal_position)
        cloned_env.agent_position = self.agent_position
        return cloned_env
    
    def get_game_over(self):
        """Détermine si l'agent a atteint la position objectif."""
        return self.agent_position == self.goal_position
