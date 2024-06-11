from player import Player
from track import Track
import random

class Game:
    """
    Implémente la logique du jeu.

    Args:
        num_players (int): Nombre de joueurs dans le jeu.
        player_colors (list): Liste des couleurs des joueurs.
        num_tracks (int): Nombre total de pistes disponibles (par défaut: 11).
        winning_tracks (int): Nombre de pistes nécessaires pour gagner (par défaut: 3).
    """

    def __init__(self, num_players, player_colors, num_tracks=11, winning_tracks=3):
        self.players = [Player(f"Player {i + 1}", player_colors[i]) for i in range(num_players)]
        track_lengths = {2: 3, 3: 5, 4: 7, 5: 9, 6: 11, 7: 13, 8: 11, 9: 9, 10: 7, 11: 5, 12: 3}
        self.tracks = [Track(i + 2, max_position=track_lengths[i + 2]) for i in range(num_tracks)]  # Tracks for numbers 2 to 12
        self.current_player_index = 0
        self.winning_tracks = winning_tracks
        self.state_vector = [(i, 0) for i in range(2, 13)]  # Initialiser le state_vector
        self.action_vector = [0 for _ in range(4)]  # Initialiser le action_vector
    
    def get_player_by_name(self, name):
        """
        Renvoie le joueur avec le nom spécifié.

        Args:
            name (str): Le nom du joueur à rechercher.

        Returns:
            Player or None: Le joueur correspondant au nom spécifié, ou None s'il n'est pas trouvé.
        """
        for player in self.players:
            if player.name == name:
                return player
        return None

    def is_player_out_of_markers(self, player):
        """
        Vérifie si le joueur a épuisé tous ses marqueurs.

        Args:
            player (Player): Le joueur à vérifier.

        Returns:
            bool: True si le joueur est à court de marqueurs, False sinon.
        """
        return player.markers == 0
    
    def update_action_vector(self, dice_decisions):
        """
        Met à jour le action_vector en fonction des décisions du joueur.

        Args:
            dice_decisions (list): Les décisions du joueur concernant les paires de dés.
        """
        self.action_vector = dice_decisions
    
    def update_state_vector(self, column, progress):
        """
        Met à jour la progression dans une colonne spécifique.

        Args:
            column (int): La colonne à mettre à jour.
            progress (int): La progression à définir pour la colonne spécifiée.
        """
        for i, (col, _) in enumerate(self.state_vector):
            if col == column:
                self.state_vector[i] = (col, progress)
                break

    def start_game(self):
        """
        Lance le jeu en alternant les tours des joueurs jusqu'à ce qu'un joueur gagne.
        """
        random_player = random.choice(self.players)  # Choisissez un joueur aléatoire
        while not self.is_game_over():
            current_player = self.players[self.current_player_index]
            self.take_player_turn(current_player, current_player == random_player)

            if self.is_game_over():
                break
            self.current_player_index = (self.current_player_index + 1) % len(self.players)

    def take_player_turn(self, current_player, is_random_player):
        """
        Gère le tour d'un joueur donné.

        Args:
            current_player (Player): Le joueur dont c'est le tour.
            is_random_player (bool): Indique si le joueur est choisi aléatoirement pour ce tour.
        """
        while current_player.markers > 0:
            dice_pairs = current_player.roll_dice()
            self.update_action_vector(dice_pairs)
            
            if is_random_player:
                self.automated_moves(current_player, dice_pairs)
                should_reroll = self.should_random_player_reroll(current_player)  
                if not should_reroll:
                    break
            else:
                self.process_moves(current_player, dice_pairs)
                if current_player.markers > 0 and input("Roll again? (y/n): ").lower() != 'y':
                    break

        current_player.secure_positions(self.tracks)
    
    def should_random_player_reroll(self, current_player):
        """
        Détermine si un joueur choisi aléatoirement doit rejouer.

        Args:
            current_player (Player): Le joueur actuel.

        Returns:
            bool: True si le joueur doit rejouer, False sinon.
        """
        return random.choice([True, False])
    
    def automated_moves(self, current_player, dice_pairs):
        """
        Effectue les mouvements automatisés pour un joueur donné.

        Args:
            current_player (Player): Le joueur actuel.
            dice_pairs (list): Les paires de dés obtenues.
        """
        available_moves = {}
        move_options = {}
        move_number = 1

        for i, pair1 in enumerate(dice_pairs):
            for j, pair2 in enumerate(dice_pairs):
                if i != j:
                    sum1 = sum(pair1)
                    sum2 = sum(pair2)
                    track_pair = tuple(sorted((sum1, sum2)))
                    if track_pair not in available_moves:
                        available_moves[track_pair] = move_number
                        move_options[str(move_number)] = track_pair
                        move_number += 1

        if move_options:
            chosen_move_number = random.choice(list(move_options.keys()))
            chosen_move_key = move_options.get(chosen_move_number)

            if chosen_move_key:
                track1, track2 = chosen_move_key
                placed1 = current_player.place_marker(self.tracks[track1 - 2])
                placed2 = current_player.place_marker(self.tracks[track2 - 2])

    def process_moves(self, current_player, dice_pairs):
        """
        Traite les mouvements pour un joueur donné.

        Args:
            current_player (Player): Le joueur actuel.
            dice_pairs (list): Les paires de dés obtenues.
        """
        available_moves = {}
        move_options = {}
        move_number = 1

        for i, pair1 in enumerate(dice_pairs):
            for j, pair2 in enumerate(dice_pairs):
                if i != j:
                    sum1 = sum(pair1)
                    sum2 = sum(pair2)
                    track_pair = tuple(sorted((sum1, sum2)))
                    if track_pair not in available_moves:
                        move_key = f"{move_number}. Move: {sum1} and {sum2} -> Tracks {track_pair[0]} and {track_pair[1]}"
                        available_moves[track_pair] = move_key
                        move_options[str(move_number)] = track_pair
                        move_number += 1

        chosen_move_number = input("Choose your move (enter a number): ")
        chosen_move_key = move_options.get(chosen_move_number)

        if chosen_move_key and chosen_move_key in available_moves:
            track1, track2 = chosen_move_key
            placed1 = current_player.place_marker(self.tracks[track1 - 2])
            placed2 = current_player.place_marker(self.tracks[track2 - 2])

    def is_game_over(self):
        """
        Vérifie si le jeu est terminé (c'est-à-dire si un joueur a gagné).

        Returns:
            bool: True si le jeu est terminé, False sinon.
        """
        for player in self.players:
            if len(player.won_tracks) >= self.winning_tracks:
                print(f"\n{player.name} has won the game!")
                return True
        return False
