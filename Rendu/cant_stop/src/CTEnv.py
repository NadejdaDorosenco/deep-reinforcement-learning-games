from typing import Tuple
import numpy as np
import random
from player import Player  
from track import Track  
import torch

class GameEnv:
    """
    Une classe pour représenter un environnement de jeu adapté pour l'apprentissage en profondeur,
    basée sur le jeu Can't Stop.
    """
    OBS_SIZE = 22 + 3  # Pour 11 pistes avec positions permanentes et 11 pistes avec positions temporaires + État supplémentaire (phase du jeu, colonnes actives, etc.)
    ACTION_SIZE = 100
    
    def __init__(self, num_players=2, player_colors=['Red', 'Blue'], num_tracks=11, winning_tracks=3):
        """
        Initialise un nouvel environnement de jeu.
        """
        self.num_players = num_players
        self.player_colors = player_colors
        self.num_tracks = num_tracks
        self.winning_tracks = winning_tracks
        self.players = [Player(f"Player {i + 1}", player_colors[i]) for i in range(num_players)]
        track_lengths = {2: 3, 3: 5, 4: 7, 5: 9, 6: 11, 7: 13, 8: 11, 9: 9, 10: 7, 11: 5, 12: 3}
        self.tracks = [Track(i + 2, max_position=track_lengths[i + 2]) for i in range(num_tracks)]
        self.current_player_index = 0
        self.phase = 0
        self.state_vector = np.zeros((self.num_players, self.OBS_SIZE))
        self.action_vector = []
        self.reset()

    def reset(self):
        """
        Réinitialise l'état de l'environnement pour un nouveau jeu.
        """
        self.players = [Player(f"Player {i + 1}", self.player_colors[i]) for i in range(self.num_players)]
        self.tracks = [Track(i + 2, max_position={2: 3, 3: 5, 4: 7, 5: 9, 6: 11, 7: 13, 8: 11, 9: 9, 10: 7, 11: 5, 12: 3}[i + 2]) for i in range(self.num_tracks)]
        self.current_player_index = random.randint(0, self.num_players - 1)
        self.phase = 0
        self.update_state_vector()
        return self.get_obs()  # Retourne l'observation initiale

    def update_state_vector(self):
        """
        Met à jour le vecteur d'état basé sur l'état actuel du jeu.
        """
        self.state_vector = np.zeros((self.num_players, self.OBS_SIZE))

        for i, player in enumerate(self.players):
            # Mise à jour des positions permanentes et temporaires sur chaque piste
            for track_idx, track in enumerate(self.tracks):
                permanent_position = track.get_player_position(player)  # Position permanente
                temporary_position = track.get_temporary_position(player)  # Position temporaire, nécessite l'implémentation dans Track

                self.state_vector[i, track_idx] = permanent_position  # Mettre à jour la position permanente
                self.state_vector[i, len(self.tracks) + track_idx] = temporary_position  # Mettre à jour la position temporaire

            # Mettre à jour les indicateurs d'état spécifiques au joueur
            self.state_vector[i, -3] = self.phase  # La phase du jeu est commune à tous les joueurs
            self.state_vector[i, -2] = player.markers  # Nombre de marqueurs restants
            self.state_vector[i, -1] = len(player.active_tracks)  # Nombre de pistes actives

    def get_obs(self) -> Tuple[float, ...]:
        """
        Retourne l'observation de l'état actuel du jeu pour le joueur actuel.
        """
        return tuple(self.state_vector[self.current_player_index])

    def step(self, action_index):
        """
        Exécute une action dans l'environnement et met à jour son état.
        """
        action = self.action_vector[action_index]
        self.take_player_turn(action)
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        self.update_state_vector()
        game_over = self.get_game_over()
        reward = self.get_score() if not game_over else 0
        return self.get_obs(), reward, game_over, {}

    def print(self):
        """
        Affiche l'état actuel du jeu dans la console.
        """
        print(f"Current player: {self.players[self.current_player_index].name}")
        for track in self.tracks:
            print(f"Track {track.number}: {track.current_position}")
    
    def available_action_mask(self):
        """
        Détermine les actions disponibles pour le joueur actuel en se basant sur l'état actuel du jeu.
        Dans ce contexte, une action disponible signifie que le joueur a la possibilité de lancer les dés et
        potentiellement de placer des marqueurs, ou de choisir d'arrêter son tour.

        :return: Un tensor PyTorch de forme (1, 2) représentant les actions disponibles (lancer les dés, arrêter le tour).
                La première valeur indique si le joueur peut continuer à lancer les dés, la deuxième s'il peut arrêter le tour.
        """
        current_player = self.players[self.current_player_index]
        actions = [0, 0]  # [Peut continuer à lancer, Peut arrêter le tour]

        # Vérifie si le joueur peut placer un marqueur temporaire sur au moins une piste
        can_place_temp_marker = any(
            track.can_place_temporary_marker(current_player) for track in self.tracks
        )
        actions[0] = 1 if can_place_temp_marker else 0

        # Vérifie si le joueur a des marqueurs temporaires qu'il peut choisir de sécuriser
        has_temporary_markers = any(
            track.get_temporary_position(current_player) > 0 for track in self.tracks
        )
        actions[1] = 1 if has_temporary_markers else 0

        # Convertir la liste Python en tensor PyTorch
        action_mask = torch.tensor(actions, dtype=torch.float)

        # Redimensionner le tensor pour avoir la forme (1, 2)
        action_mask = action_mask.unsqueeze(0)

        return action_mask
    
    def get_score(self) -> float:
        """
        Calcule et retourne la récompense (score) pour le joueur actuel en fonction des actions et des événements
        survenus pendant son tour.

        :return: La récompense calculée pour le joueur actuel.
        """
        current_player = self.players[self.current_player_index]
        reward = 0.0

        # Récompenser chaque piste gagnée pendant le tour
        reward += len(current_player.won_tracks) * 1.0

        # Encourager le placement de marqueurs temporaires
        # Supposons que chaque marqueur temporaire placé donne un petit bonus
        # Cela encourage les joueurs à avancer même s'ils n'ont pas sécurisé la piste
        for track in self.tracks:
            if track.get_temporary_position(current_player) > 0:
                reward += 0.1

        # Récompenser la sécurisation des progrès
        # Un bonus plus important pour chaque piste où le joueur a sécurisé sa position
        for track in self.tracks:
            if track.get_player_position(current_player) > 0 and track.get_player_position(current_player) == track.get_temporary_position(current_player):
                reward += 0.5

        # Réinitialiser les indicateurs pour le prochain tour du joueur
        current_player.reset_for_next_turn()

        return reward

    def get_game_over(self) -> bool:
        """
        Détermine si le jeu est terminé.

        :return: True si le jeu est terminé, False sinon.
        """
        for player in self.players:
            if len(player.won_tracks) >= self.winning_tracks:
                return True
        return False
    
    def take_player_turn(self, action):
        """
        Traite le tour d'un joueur en fonction de l'action choisie.

        :param action: L'action choisie par le joueur, représentée par une liste [dice_combination, continue_indicator].
                    Pour continue_indicator, 0 signifie "arrêter le tour" et 1 "continuer à lancer les dés".
                    dice_combination représente le move choisi sous forme de tuple (sum1, sum2).
        """
        current_player = self.players[self.current_player_index]
        dice_combination, continue_indicator = action

        if continue_indicator == 0:
            # Arrêter le tour et sécuriser les positions si l'action est 0
            current_player.secure_positions(self.tracks)
            #print(f"{current_player.name} a choisi de sécuriser ses positions et de terminer son tour.")
        else:
            # Continuer avec la combinaison de dés spécifiée par l'action
            sum1, sum2 = dice_combination
            # Vérifie et place des marqueurs sur les pistes correspondantes à sum1 et sum2
            placed1 = current_player.place_marker(self.tracks[sum1 - 2])
            placed2 = current_player.place_marker(self.tracks[sum2 - 2])
            """
            if placed1:
                print(f"{current_player.name} a placé un marqueur sur la piste {sum1}")
            elif not placed1: 
                print(f"{current_player.name} n'a pas pu placer un marqueur sur la piste {sum1}")
            elif placed2:
                print(f"{current_player.name} a placé un marqueur sur la piste {sum2}.")
            else:
                print(f"{current_player.name} n'a pas pu placer un marqueur sur la piste {sum2}")
            """
            # Mise à jour de l'état du jeu en fonction de l'action effectuée
            self.update_state_vector()
            

    def get_dice_combinations(self, dice_results):
        """
        Génère toutes les combinaisons de dés possibles à partir des résultats des dés.
        :param dice_results: Un tuple de résultats de lancer de dés.
        :return: Une liste de tuples représentant toutes les combinaisons valides de dés.
        """
        current_player = self.players[self.current_player_index]
        available_moves = []
        for pair in dice_results:
            sum1, sum2 = pair[0], pair[1]
            if current_player.can_place_marker(self.tracks[sum1 - 2]) or current_player.can_place_marker(self.tracks[sum2 - 2]):
                available_moves.append((sum1,sum2))
        return available_moves  
    
    def generate_action_vector(self, dice_results):
        """
        Génère l'action_vector basé sur les résultats actuels des dés.
        :param dice_results: Les résultats du lancer de dés actuel.
        """
        self.action_vector = []
        # Ajouter des actions pour chaque combinaison possible de dés
        for dice_combination in self.get_dice_combinations(dice_results):
            self.action_vector.append([dice_combination, 1])  # Continuer avec cette combinaison
        # Ajouter une option pour arrêter le tour
        self.action_vector.append([(0, 0), 0])  # (0, 0) est un placeholder pour "pas de mouvement"

    def clone_stochastic(self):
        """
        Crée une copie stochastique de l'environnement de jeu, y compris les états des joueurs et des pistes.
        Cette méthode permet de simuler des scénarios futurs sans affecter l'état actuel du jeu.
        """
        # Création d'une nouvelle instance de l'environnement de jeu
        cloned_env = GameEnv(self.num_players, self.player_colors, self.num_tracks, self.winning_tracks)
        
        # Clonage des joueurs
        # Assurez-vous que la méthode clone() dans la classe Player copie fidèlement l'état du joueur,
        # y compris les marqueurs temporaires et permanents, les pistes actives, etc.
        cloned_env.players = [player.clone() for player in self.players]
        
        # Clonage des pistes
        # La méthode clone() dans la classe Track devrait copier l'état de chaque piste,
        # y compris les positions des joueurs (à la fois temporaires et permanentes) et l'état de la piste (par exemple, si elle est gagnée).
        cloned_env.tracks = [track.clone() for track in self.tracks]
        
        # Copie des autres attributs d'état pertinents
        cloned_env.current_player_index = self.current_player_index
        cloned_env.phase = self.phase
        # Assurez-vous que le vecteur d'état est correctement copié, en tenant compte de la nouvelle structure si nécessaire
        cloned_env.state_vector = np.copy(self.state_vector)
        # Copie de l'action_vector si nécessaire
        cloned_env.action_vector = list(self.action_vector)
        
        return cloned_env
    
    def roll_dice_and_generate_actions(self):
        """
        Lance les dés pour le joueur actuel et génère les actions possibles basées sur les résultats.
        
        Cette méthode effectue deux opérations principales :
        1. Elle utilise la méthode `roll_dice` de l'instance du joueur actuel pour simuler le lancer de quatre dés,
        produisant ainsi un ensemble de paires de dés possibles.
        2. Elle appelle ensuite `generate_action_vector` en passant les résultats des dés pour construire l'`action_vector`,
        qui contient toutes les actions possibles (combinaisons de dés et indicateurs de continuation) que le joueur peut
        choisir pour son tour actuel.
        
        L'`action_vector` généré inclut à la fois les actions pour avancer sur les pistes en fonction des combinaisons de dés
        et l'action pour arrêter le tour, permettant ainsi une prise de décision stratégique basée sur les résultats des dés.
        
        :param None: Cette méthode ne prend aucun paramètre.
        :return: None. Les résultats des dés et l'`action_vector` sont gérés à l'intérieur de la méthode.
        """
        current_player = self.players[self.current_player_index]
        dice_results = current_player.roll_dice()  # Obtention des résultats des dés via la méthode du joueur
        # Utiliser get_dice_combinations pour filtrer les combinaisons valides
        valid_combinations = self.get_dice_combinations(dice_results)
        self.generate_action_vector(valid_combinations)  # Génération de l'`action_vector` basé sur les résultats des dés