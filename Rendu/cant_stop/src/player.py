import random

class Player:
    """Représente un joueur dans le jeu."""
    def __init__(self, name, color):
        """
        Initialise un joueur avec son nom et sa couleur.

        Args:
            name (str): Le nom du joueur.
            color (str): La couleur associée au joueur.
        """
        self.name = name
        self.color = color
        self.markers = 3  # Marqueurs disponibles pour jouer
        self.active_tracks = []  # Pistes actives pour ce tour
        self.won_tracks = []  # Pistes gagnées
        self.temp_positions = {}  # Nouveau: positions temporaires sur les pistes pendant un tour
        
        self.placed_marker_this_turn = False
        self.secured_advance_this_turn = False
        
    def clone(self):
        """
        Crée une copie profonde de cette instance de joueur.

        Returns:
            Player: Une copie du joueur.
        """
        # Crée une copie profonde de cette instance
        cloned_player = Player(self.name, self.color)
        cloned_player.markers = self.markers
        cloned_player.active_tracks = list(self.active_tracks)
        cloned_player.won_tracks = list(self.won_tracks)
        return cloned_player

    def has_won(self):
        """
        Vérifie si le joueur a gagné en atteignant le sommet de 3 colonnes différentes.

        Returns:
            bool: True si le joueur a gagné, False sinon.
        """
        # Vérifie si le joueur a atteint le sommet de 3 colonnes différentes
        # Cette vérification dépend de la façon dont vous stockez l'état du jeu dans le joueur
        # Par exemple, si vous avez une liste ou un ensemble de colonnes gagnées :
        return len(self.won_tracks) >= 3

    def can_place_marker(self, track):
        """
        Vérifie si le joueur peut placer un marqueur sur la piste donnée.

        Args:
            track (Track): La piste sur laquelle vérifier si le marqueur peut être placé.

        Returns:
            bool: True si le joueur peut placer un marqueur, False sinon.
        """
        # Vérifie si le joueur peut placer un marqueur sur la piste
        return (self.markers > 0 and 
                (track.number in self.active_tracks or len(self.active_tracks) < 3) and
                not track.won)

    def place_marker(self, track):
        """
        Place un marqueur sur la piste donnée si possible.

        Args:
            track (Track): La piste sur laquelle placer le marqueur.

        Returns:
            bool: True si le marqueur a été placé avec succès, False sinon.
        """
        # Place un marqueur sur la piste si possible
        if self.can_place_marker(track):
            self.markers -= 1
            if track.number not in self.active_tracks:
                self.active_tracks.append(track.number)
            track.add_temp_marker(self)  # Ajouter un marqueur temporaire
            self.placed_marker_this_turn = True # Indicateur pour suivre si un marqueur a été placé mis à jour
            return True
        return False

    def secure_positions(self, tracks):
        """
        Sécurise les positions des marqueurs en pions permanents sur les pistes spécifiées.

        Args:
            tracks (list): Liste des pistes à vérifier et à sécuriser.
        """
        advances_secured = False
        for track in tracks:
            if track.number in self.active_tracks and not track.won:
                if track.move_bonze(self, 1):  # Avancer d'une case sur la piste
                    if track.get_player_position(self) == track.max_position:
                        advances_secured = True
                        if track.get_player_position(self) == track.max_position:
                            self.won_tracks.append(track)
                            track.win_track(self)
        self.secured_advance_this_turn = advances_secured  # Indicateur pour suivre les avances sécurisées mis à jour
        self.reset_markers()  # Réinitialiser les marqueurs pour le prochain tour
        
    def reset_for_next_turn(self):
        """Réinitialise les indicateurs pour le prochain tour."""
        # Réinitialiser les indicateurs pour le prochain tour
        self.placed_marker_this_turn = False
        self.secured_advance_this_turn = False

    def reset_markers(self):
        """Réinitialise les marqueurs pour un nouveau tour."""
        # Réinitialise les marqueurs pour un nouveau tour
        self.markers = 3
        self.placed_marker_this_turn = False
        self.secured_advance_this_turn = False
        self.active_tracks = []
        self.temp_positions = {}

    def roll_dice(self):
        """
        Lance les dés et génère des paires de valeurs.

        Returns:
            list: Liste des paires de valeurs de dés générées.
        """
        # Lance les dés et générer des paires
        dice_values = [random.randint(1, 6) for _ in range(4)]
        dice_pairs = set()
        for i in range(4):
            for j in range(i + 1, 4):
                dice_pairs.add((dice_values[i], dice_values[j]))
        return list(dice_pairs)

    def calculate_moves(self, dice_pairs, tracks):
        """
        Calcule les mouvements possibles basés sur les paires de valeurs de dés et les pistes disponibles.

        Args:
            dice_pairs (list): Liste des paires de valeurs de dés.
            tracks (list): Liste des pistes disponibles pour le mouvement.

        Returns:
            dict: Un dictionnaire où les clés sont les numéros de pistes et les valeurs sont les paires de dés correspondantes.
        """
        moves = {}
        for d1, d2 in dice_pairs:
            track_num = d1 + d2
            if any(track.number == track_num for track in tracks):
                moves.setdefault(track_num, []).append((d1, d2))
        return moves
