import random, time
import numpy as np

class Dice:
    """Represents a dice with different faces."""
    
    def __init__(self, faces):
        self.faces = faces
        
    def roll(self):
        """Rolls the dice and returns a random face."""
        return random.choice(self.faces)

class BalloonGameSolo:
    def __init__(self):
        self.columns = {
            "Ballon jaune": [0, 3, 7, 11, 15, 3],
            "Ballon bleu": [1, 3, 5, 7, 9, 12, 8],
            "Ballon rouge": [0, 0, 0, 2, 4, 6, 8, 10, 14, 6],
            "Etoile": [1, 2, 3, 5, 7, 10, 13, 16, 4],
            "Lune": [2, 3, 4, 5, 7, 9, 12, 5],
            "Losange": [1, 3, 6, 10, 13, 7]
        }

        self.all_possible_faces = ["Rouge etoile", "Rouge lune", "Rouge losange", "Bleu etoile", "Bleu lune", "Jaune etoile"]
        self.original_dice_set = [Dice(self.all_possible_faces) for _ in range(3)]  
        self.dice_set = self.original_dice_set.copy()
        self.rolls = []
        # Initialize progression to -1 for each column
        self.progression = {key: -1 for key in self.columns}
        self.actual_progression = {key: -1 for key in self.columns}

        self.break_count = 0  # Compte les breaks
        self.break_scores = []  # Stocke les scores à chaque break
        self.broken_columns = set()  # Un ensemble pour stocker les colonnes qui ont déjà connu un break.
        
    # version random : A REFAIRE AVEC LES NOUVELLES ACTIONS
    def roll_dice(self, reroll_indices=[]):
        """Rolls all dice and stores the results. Can reroll specific dice if indices are given."""
        results = []

        # S'assurer que self.rolls et self.dice_set ont la même longueur
        while len(self.rolls) < len(self.dice_set):
            self.rolls.append(None)

        if reroll_indices:  # Si il y a des indices à relancer
            for i in reroll_indices:
                if i < len(self.dice_set):  # Vérifier si l'indice est dans la plage
                    self.rolls[i] = self.dice_set[i].roll()  # Relancer les dés aux indices spécifiés
                    results.append(self.rolls[i])
                else:
                    # Gérer le cas où l'indice est hors plage
                    print(f"Warning: Index {i} out of range for dice_set")

            # Ajouter un dé supplémentaire
            if len(self.rolls) < 5:
                extra_dice = Dice(self.dice_set[-1].faces)
                extra_result = extra_dice.roll()
                self.rolls.append(extra_result)
                results.append(extra_result)
                self.dice_set.append(extra_dice)

        else:  # Si il n'y a pas d'indices à relancer
            self.dice_set = self.original_dice_set.copy()
            for dice in self.dice_set:
                result = dice.roll()
                results.append(result)
            self.rolls = results
        return results

    def record_scores(self):
        """Records the scores based on the dice rolls."""
        for roll in self.rolls:
            # Vérifier que le roll n'est pas None
            if roll is not None:
                color, shape = roll.split()
                for column, scores in self.columns.items():
                    if color.lower() in column.lower() or shape.lower() in column.lower():
                        # Check the next progression score
                        if self.progression[column] < len(scores) - 1:
                            self.progression[column] += 1
            else :
                print("None")

    
    def record_actual_score(self):
        # Réinitialiser actual_progression à la valeur de progression pour le début de chaque appel
        self.actual_progression = self.progression.copy()

        for roll in self.rolls:
            color, shape = roll.split()
            for column, scores in self.columns.items():
                if color.lower() in column.lower() or shape.lower() in column.lower():
                    # Utiliser une variable temporaire pour suivre la progression actuelle de la colonne
                    current_progress = self.actual_progression[column]

                    # Vérifier si nous pouvons mettre à jour la progression
                    if current_progress < len(scores) - 1:
                        self.actual_progression[column] = current_progress + 1


    def get_game_state_vector(self):
        state_vector = []

        # Encodage des faces des dés
        for roll in self.rolls:
            if roll is not None:
                index = self.all_possible_faces.index(roll)
            state_vector.append(index)
        # Padding pour les dés non lancés
        num_dice_not_rolled = 5 - len(self.rolls)  # Calculer le nombre de dés non lancés
        for _ in range(num_dice_not_rolled):
            state_vector.append(-1)  # Ajouter le padding pour chaque dé non lancé
        # Nombre de relances
        nb_relances = 5 - len(self.rolls)
        state_vector.append(nb_relances)

        # Progression dans chaque colonne
        #self.record_actual_score()
        # Score actuel dans chaque colonne
        for column in self.columns.keys():
            progression = self.progression[column]
            #score = self.columns[column][progression] if progression > -1 else 0
            state_vector.append(progression)

        # Nombre de breaks
        state_vector.append(self.break_count)  # Ajouter le nombre de 'Breaks' déclenchés
        # Scores des breaks
        break_scores = self.break_scores + [0] * (3 - len(self.break_scores))
        state_vector.extend(break_scores)
        return np.array(state_vector)
    
    def reward(self):
        reward = 0
        for column in self.columns.keys():
            actual_progression = self.actual_progression[column]
            actual_score = self.columns[column][actual_progression] if actual_progression > -1 else 0
            reward += actual_score
        return reward

    # Doit être appelé que si relance possible (pas + que 2)
    '''def get_action_vector(self, reroll_indices):
        action_vector = [0,0,0,0] # Initialisez un vecteur de longueur 4, on ne peut pas relancer le dernier dé dans tous les cas (si on arrive à 5 dés on passe au tour suivant)
        for index in reroll_indices:
            action_vector[index] = 1  # Marquez les dés à relancer comme 1 dans le vecteur d'action.
        return np.array(action_vector)  # Retourne une liste de booléens représentant les dés relancés.'''

    def calculate_score(self):
        # calcule le score du break
        score = 0
        for column, progression in self.progression.items():
            if progression >= 0:
                score += self.columns[column][progression]
        return score
    
    def check_for_breaks(self):
        for column, scores in self.columns.items():
            if column not in self.broken_columns and self.progression[column] == len(scores) - 1:
                #print(f"Break dans la colonne {column} !")
                self.break_count += 1
                self.broken_columns.add(column)
                self.break_scores.append(self.calculate_score())
                #print(f"Score du Break {self.break_count}: {self.calculate_score()}")
                #if self.break_count >= 3:
                    #print("Fin du jeu atteinte.")
                return True # Un break s'est produit
        return False
    

    def reset(self):
        """Resets the game to the initial state."""
        self.progression = {key: -1 for key in self.columns}
        self.actual_progression = self.progression.copy()
        self.break_count = 0
        self.break_scores = []
        self.broken_columns = set()
        self.rolls = []
        self.dice_set = self.original_dice_set.copy()
        self.roll_dice()
        return self.get_game_state_vector()

    #step random
    '''def step(self, action):
        # Vérifier si l'action est de ne pas relancer (tous les éléments sont 0, sauf le dernier qui est toujours 0)
        if action == []:
            self.record_scores()
            self.rolls = []
            self.dice_set = self.original_dice_set.copy()
            self.roll_dice()
            self.num_dice = len(self.dice_set)
            is_break = self.check_for_breaks()
            done = False
            if self.break_count == 3:
                done = True
            next_state = self.get_game_state_vector()
        else:
            self.roll_dice(action)
            is_break = False
            done = False
            if len(self.rolls) == 5:
                #print(self.get_game_state_vector())
                self.record_scores()
                self.rolls = []
                self.dice_set = self.original_dice_set.copy()
                self.roll_dice() # ajout
                self.num_dice = len(self.dice_set)
                is_break = self.check_for_breaks() # ajout
                done = False # ajout
                if self.break_count == 3: # ajout
                    done = True # ajout
            next_state = self.get_game_state_vector()
        return next_state, is_break, done'''
    
    def step(self):
        self.record_scores()
        reward = self.calculate_score()
        next_state = self.get_game_state_vector()
        done = False 
        self.check_for_breaks()
        if self.break_count == 3: 
            done = True
        return reward, next_state, done

    def play_random_turn(self):
        """Simule un tour avec des choix aléatoires pour les relances."""
        self.rolls = []
        self.dice_set = self.original_dice_set.copy()
        
        #print("\nRésultats du lancer initial:")
        self.roll_dice()
        current_state = self.get_game_state_vector()
        #print(current_state) #TEST
        for _ in range(2):  # Deux relances possibles
            reroll_choice = random.choice([True, False])
            #print("\nVoulez-vous relancer certains dés ? (Oui/Non):", "Oui" if reroll_choice else "Non")
            if reroll_choice:
                reroll_indices = sorted(random.sample(range(len(self.rolls)), random.randint(1, len(self.rolls))))
                #print("Quels dés voulez-vous relancer? Séparez par des virgules:", reroll_indices)
                #print("\nRésultats du relance:")
                #action = self.get_action_vector(reroll_indices)
                #print("action: ", action) #TEST
                #print("reroll_indices: ", reroll_indices) #TEST
                self.roll_dice(reroll_indices)
                #print("\nRésultats du lancer: ")
                current_state = self.get_game_state_vector()
                #print(current_state) #TEST
                #print(self.rolls)
            else:
                #action = self.get_action_vector([])
                #print("action: ", action) #TEST
                #print("\nRésultat: ")
                current_state = self.get_game_state_vector()
                #print(current_state) #TEST
                break

        self.record_scores()
        #print("\nScores actuels:")
        for column, progression in self.progression.items():
            score = self.columns[column][progression] if progression >= 0 else 0
            #print(f"{column}: {score}")


    def simulate_game(self):
        """Simule un jeu jusqu'à la condition de fin."""
        rounds = 1
        start_time = time.time()
        while self.break_count < 3:
            #print(f"\n--- Tour {rounds} ---")
            self.play_random_turn()
            balloon_popped = False
            for column, scores in self.columns.items():
                if column not in self.broken_columns and self.progression[column] == len(scores) - 1:
                    #print(f"\nBreak dans la colonne {column} !")
                    balloon_popped = True
                    self.broken_columns.add(column)
                    break

            if balloon_popped:
                self.break_count += 1
                current_score = self.calculate_score()
                self.break_scores.append(current_score)
                #print(f"Score du Break {self.break_count}: {current_score}")

            rounds += 1

        total_score = sum(self.break_scores)
        #elapsed_time = time.time() - start_time
        #print(f"\nFin du jeu ! Score total: {total_score}, , Tours joués: {rounds}")
        return total_score, rounds
