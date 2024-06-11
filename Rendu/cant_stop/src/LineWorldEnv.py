import numpy as np

class LineWorldEnv:
    """
    Un environnement simple 'Line World' pour l'apprentissage par renforcement.
    L'agent se déplace dans un monde linéaire composé d'une série d'états, avec deux états terminaux aux extrémités.
    """
    ACTION_SIZE = 2  # Définit le nombre d'actions possibles : gauche (-1) et droite (+1)

    def __init__(self, size=5):
        self.size = size  # Définit la taille du monde linéaire
        self.state = np.array([1])    # Position initiale de l'agent (en partant de 0)
        self.done = False # Indique si l'épisode est terminé

    def step(self, action):
        """
        Effectue une action dans l'environnement.
        
        :param action: -1 pour un déplacement vers la gauche, +1 pour un déplacement vers la droite
        :return: (new_state, reward, done)
        """
        if self.done:
            return self.state, 0, self.done

        # Mettre à jour l'état basé sur l'action
        self.state += action
        
        # Vérifier les conditions terminales
        if self.state <= 0 or self.state >= self.size - 1:
            self.done = True
            reward = 1 if self.state >= self.size - 1 else -1
        else:
            reward = 0

        return self.state, reward, self.done, {}

    def reset(self):
        """
        Réinitialise l'environnement à son état initial.
        """
        self.state = np.array([1])
        self.done = False
        return self.state

    def render(self):
        """
        Affiche l'état actuel de l'environnement.
        """
        environment = ['-' for _ in range(self.size)]
        environment[self.state[0]] = 'A'  # 'A' pour Agent
        print(''.join(environment))
        
    def get_game_over(self):
        """
        Retourne True si l'épisode est terminé, sinon False.
        """
        return self.done
    
    def clone_stochastic(self):
        """
        Retourne une copie profonde de cet environnement pour la simulation.
        """
        cloned_env = LineWorldEnv(self.size)
        cloned_env.state = np.copy(self.state)
        cloned_env.done = self.done
        return cloned_env
