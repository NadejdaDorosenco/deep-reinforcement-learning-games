class Track:
    """Représente une piste dans le jeu."""
    def __init__(self, number, max_position):
        """
        Initialise une piste.

        Args:
            number (int): Le numéro de la piste.
            max_position (int): La position maximale sur la piste.
        """
        self.number = number
        self.max_position = max_position
        self.positions = {}  # dictionnaire pour suivre la progression des joueurs avec des pions permanents
        self.won = False
        self.temp_positions = {}  # Nouveau: positions temporaires des joueurs
    
    def clone(self):
        """
        Crée une copie profonde de cette piste.

        Returns:
            Track: La copie de la piste.
        """
        # Crée une copie profonde de cette instance
        cloned_track = Track(self.number, self.max_position)
        return cloned_track
    
    def get_temporary_position(self, player):
        """
        Retourne la position temporaire du joueur sur la piste.

        Args:
            player: Le joueur dont on veut connaître la position temporaire.

        Returns:
            int: La position temporaire du joueur sur la piste.
        """
        # Retourne la position temporaire du joueur sur la piste, ou 0 si non présente
        return self.temp_positions.get(player, 0)

    def to_int(self):
        """
        Convertit le numéro de la piste en entier.

        Returns:
            int: Le numéro de la piste.
        """
        return self.number
    
    def can_place_temporary_marker(self, player):
        """
        Vérifie si le joueur peut placer un marqueur temporaire sur cette piste.

        Args:
            player: Le joueur souhaitant placer un marqueur temporaire.

        Returns:
            bool: True si le joueur peut placer un marqueur temporaire, False sinon.
        """
        # Vérifie si le joueur peut placer un marqueur temporaire sur cette piste
        return player not in self.temp_positions
    
    def add_temp_marker(self, player):
        """
        Ajoute un marqueur temporaire pour le joueur sur cette piste.

        Args:
            player: Le joueur pour lequel ajouter un marqueur temporaire.
        """
        # Ajoute un marqueur temporaire pour le joueur
        if player not in self.temp_positions:
            self.temp_positions[player] = 1  # Commencer avec une position temporaire
        else:
            self.temp_positions[player] += 1  # Avancer le marqueur temporaire

    def add_permanent_marker(self, player):
        """
        Ajoute un marqueur permanent pour le joueur sur cette piste.

        Args:
            player: Le joueur pour lequel ajouter un marqueur permanent.
        """
        if player not in self.positions:
            #print("Player ", player.name, " not in this positions; positionning at 0")
            self.positions[player] = 0
            
    def convert_temp_to_permanent(self, player):
        """
        Convertit les marqueurs temporaires du joueur en marqueurs permanents.

        Args:
            player: Le joueur pour lequel convertir les marqueurs.
        """
        # Convertit les marqueurs temporaires en permanents pour le joueur
        if player in self.temp_positions:
            if player not in self.positions:
                self.positions[player] = self.temp_positions[player]
            else:
                self.positions[player] = max(self.positions[player], self.temp_positions[player])
            del self.temp_positions[player]  # Supprimer les positions temporaires après conversion

    def reset_temp_markers(self):
        """Réinitialise les marqueurs temporaires sur cette piste."""
        self.temp_positions = {}

    def move_bonze(self, player, steps):
        """
        Déplace le marqueur du joueur sur la piste.

        Args:
            player: Le joueur dont le marqueur doit être déplacé.
            steps (int): Le nombre de pas à avancer.

        Returns:
            bool: True si le déplacement a réussi, False sinon.
        """
        if not self.won and player in self.positions:
            new_position = self.positions[player] + steps
            #print("Moved from position ", self.positions[player], "to ", new_position)
            #print("Reminder, maxposition is ", self.max_position)
            if new_position <= self.max_position:
                self.positions[player] = new_position
                if new_position == self.max_position:
                    self.win_track(player)
                return True
            #print("won ? ", self.won)
        return False

    def is_won_by_player(self, player):
        """
        Vérifie si la piste a été remportée par un joueur spécifique.

        Args:
            player: Le joueur dont on veut vérifier s'il a remporté la piste.

        Returns:
            bool: True si le joueur a remporté la piste, False sinon.
        """
        #print("Player ", player.name, " won track ", self.number)
        return self.won and player in self.positions and self.positions[player] == self.max_position

    def win_track(self, winner):
        """
        Marque la piste comme remportée par un joueur spécifique.

        Args:
            winner: Le joueur ayant remporté la piste.
        """
        if not self.won:
            self.won = True
        # Retirer les pions des autres joueurs
        for player in list(self.positions):
            if player != winner:
                del self.positions[player]
        #print("Player ", winner.name, " won track ", self.number)

    def get_player_position(self, player):
        """
        Obtient la position actuelle du joueur sur la piste.

        Args:
            player: Le joueur dont on veut connaître la position.

        Returns:
            int: La position du joueur sur la piste.
        """
        return self.positions.get(player, 0)