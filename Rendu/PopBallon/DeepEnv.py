from model import BalloonGameSolo, Dice

class DeepEnv:
    OBS_SIZE = 16  # longueur du vecteur d'état
    ACTION_SIZE = 16  # nombre de dés à relancer

    def __init__(self, balloon_game):
        self.balloon_game = balloon_game  # Instance de votre jeu Balloon Pop
    
    def reset(self):
        return self.balloon_game.reset()
    
    def get_rolls(self):
        num_dice_in_play = len(self.balloon_game.rolls)
        return num_dice_in_play
    
    def check_for_breaks(self):
        return self.balloon_game.check_for_breaks()
    
    def print_env(self):  # renamed to avoid conflict with the built-in print
        print(self.balloon_game.get_game_state_vector())

    def roll_dice(self, reroll_indices=[]):
        return self.balloon_game.roll_dice(reroll_indices)
    
    def step(self):
        """Effectue une action dans l'environnement et retourne le nouvel état, la récompense, et si le jeu est terminé."""
        return self.balloon_game.step()

    # def available_actions_mask(self) -> List[bool]:
    def available_actions_mask(self):
        # Le nombre de dés actuellement en jeu
        num_dice_in_play = len(self.balloon_game.rolls)

        if num_dice_in_play > 3:
            # Créer un masque avec True pour les dés en jeu
            mask = [True for _ in range(self.ACTION_SIZE)]

        # définir les actions à False pour le 4ème dé s'il n'est pas encore en jeu
        else:
            mask = [True for _ in range(8)]
            mask.extend([False for _ in range(8)])

        return mask
        
    # Renvoie le score de chaquz break : 
    # bool = True si break vient d'arriver
    def get_final_score(self):
        # La récompense est simplement la somme des scores obtenus lors des breaks
        return sum(self.balloon_game.break_scores)
    
    '''
    Score testé dans une ancienne version qui donnait 0 sauf en fin de partie où le reward était le score final du jeu
    Fonction abandonné mais qu'on aurait du garder 
    def get_score(self,bool):
        # La récompense est simplement la somme des scores obtenus lors des breaks
        if bool:
            nb_breaks = self.balloon_game.break_count
            score = self.balloon_game.break_scores[nb_breaks -1]
            return score
        else:
            return 0
    '''
    
    def get_game_over(self):
        if self.balloon_game.break_count == 3:
            return True
        return False

    def get_game_state_vector(self):
        return self.balloon_game.get_game_state_vector()

    def get_reroll_indices(self, action):
        if action == 0:     # Conserver résultat
            return []
        elif action == 1:   # Relancer dé 1
            return [0]

        elif action == 2:   # Relancer dé 2
            return [1]  

        elif action == 3:   # Relancer dés 1 et 2
            return [0,1]

        elif action == 4:   # Relancer dé 3
            return [2]   

        elif action == 5:   # Relancer dés 1 et 3
            return [0,2]

        elif action == 6:   # Relancer dés 2 et 3
            return [1,2]

        elif action == 7:   # Relancer dés 1, 2 et 3
            return [0,1,2]

        elif action == 8:   # Relancer dé 4 
            return [3]

        elif action == 9:   # Relancer dés 1 et 4
            return [0,3]

        elif action == 10:   # Relancer dés 2 dé 4
            return [1,3]  

        elif action == 11:  # Relancer dés 1, 2 et 4
            return [0,1,3]

        elif action == 12:  # Relancer dés 3 et 4
                return [2,3]

        elif action == 13:  # Relancer dés 1, 3 et 4
            return [0,2,3]

        elif action == 14:  # Relancer dés 2, 3 et 4
            return [1,2,3] 

        elif action == 15:  # Relancer dés 1, 2, 3 et 4
            return [0,1,2,3]  
        

    def clone_stochastic(self): 
        cloned_game = BalloonGameSolo() 
        # Réassigner l'état initial des colonnes, car elles sont statiques 
        cloned_game.columns = { 
            "Ballon jaune": [0, 3, 7, 11, 15, 3], 
            "Ballon bleu": [1, 3, 5, 7, 9, 12, 8], 
            "Ballon rouge": [0, 0, 0, 2, 4, 6, 8, 10, 14, 6], 
            "Etoile": [1, 2, 3, 5, 7, 10, 13, 16, 4], 
            "Lune": [2, 3, 4, 5, 7, 9, 12, 5], 
            "Losange": [1, 3, 6, 10, 13, 7] 
        } 
        
        cloned_game.all_possible_faces = ["Rouge etoile", "Rouge lune", "Rouge losange", "Bleu etoile", "Bleu lune", "Jaune etoile"] 
        cloned_game.original_dice_set = [Dice(dice.faces) for dice in self.balloon_game.original_dice_set] 
        cloned_game.dice_set = [Dice(dice.faces) for dice in self.balloon_game.original_dice_set] 
        cloned_game.rolls = self.balloon_game.rolls[:] 
        cloned_game.progression = self.balloon_game.progression.copy() 
        cloned_game.actual_progression = self.balloon_game.actual_progression.copy() 
        cloned_game.break_count = self.balloon_game.break_count 
        cloned_game.break_scores = self.balloon_game.break_scores[:] 
        cloned_game.broken_columns = self.balloon_game.broken_columns.copy() 

        # Retourner une nouvelle instance de DeepEnv avec le jeu cloné 
        return DeepEnv(cloned_game)