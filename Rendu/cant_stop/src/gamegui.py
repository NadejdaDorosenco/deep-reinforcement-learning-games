import tkinter as tk
from game import Game
from tkinter import messagebox
import random
from time import sleep

class GameGUI:
    def __init__(self, master, num_players):
        self.master = master
        self.num_players = num_players
        self.is_running = True  # Nouvel attribut pour suivre l'état de l'application
        self.game_over = False
        master.protocol("WM_DELETE_WINDOW", self.on_close)
        master.title("Can't Stop Board")
        self.player_colors = ["red", "blue", "green", "yellow"][:num_players]
        self.game = Game(num_players, self.player_colors)
        self.canvas = tk.Canvas(master, width=600, height=400, bg='white')
        self.canvas.pack()
        self.display_player_colors()
        self.draw_tracks()
        self.roll_dice_button = tk.Button(master, text="Roll Dice", command=self.roll_dice)
        self.roll_dice_button.pack()
        self.dice_result_label = tk.Label(master, text="")
        self.dice_result_label.pack()
        self.chosen_move_label = tk.Label(master, text="")
        self.chosen_move_label.pack()
        self.chosen_move = None
        self.move_buttons = []
        self.current_turn_markers = []
        self.decision_frame_created = False
        self.player_track_positions = {player.name: {i: 0 for i in range(2, 13)} for player in self.game.players}
        self.player_markers = {player.name: [None] * 12 for player in self.game.players}
        self.active_tracks = set()
        self.won_columns = {}
        self.track_lengths = {2: 3, 3: 5, 4: 7, 5: 9, 6: 11, 7: 13, 8: 11, 9: 9, 10: 7, 11: 5, 12: 3}
        self.state_vector = [(i, 0) for i in range(2, 13)]
        self.action_vector = [0 for _ in range(4)]
        self.decision_frame = None
        self.previous_permanent_positions = {player.name: {i: 0 for i in range(2, 13)} for player in self.game.players}
        self.track_text_heights = {i: 30 for i in range(2, 13)}
        self.auto_player = random.choice(self.game.players)  # Choose an auto player at random
        self.continue_playing_auto = True
        self.init_state_vector()

    def init_state_vector(self):
        self.state_vector = {
            'game_phase': 1,  # Commencer par la phase de décision (1) - 0 pour "choisir de s'arrêter ou continuer", 1 pour "choisir une paire de dés"
            'player_positions': [[0 for _ in range(11)] for _ in range(self.num_players)], # Liste des positions du joueur actuel sur chaque colonne - 11 colonnes pour chaque joueur
            'active_columns': [0, 0, 0],  # Colonnes actives pour le tour actuel - Aucune colonne active au début
        }
        self.update_state_vector()

    def update_state_vector(self):
        # Mise à jour des positions des joueurs
        for i, player in enumerate(self.game.players):
            for track in range(2, 13):
                self.state_vector['player_positions'][i][track-2] = self.player_track_positions[player.name][track]

    def update_action_vector(self, dice_decisions):
        # Mettre à jour le action_vector en fonction des décisions du joueur
        self.action_vector = dice_decisions
        
    def display_player_colors(self):
        # Vérifiez si le jeu est terminé ou si la fenêtre principale n'existe plus
        if not self.is_running or not self.master.winfo_exists():
            return  # Sortir de la fonction si l'application n'est plus en cours d'exécution ou si la fenêtre principale n'existe plus
        # Position de départ pour l'affichage des couleurs des deux premiers joueurs
        y1 = 360  
        # Position de départ pour l'affichage des couleurs des joueurs 3 et 4
        y2 = 360  
        radius = 10  # Rayon des jetons de couleur

        for i, color in enumerate(self.player_colors):
            # Coordonnées du jeton pour les joueurs 1 et 2
            x1, x2 = 80 - radius, 80 + radius

            # Ajuster les coordonnées pour les joueurs 3 et 4
            if self.num_players > 2 and i >= 2:
                x1, x2 = 300 - radius, 300 + radius  # Décaler vers la droite
                y = y2
                y2 += 30  # Augmenter y2 pour le prochain joueur
            else:
                y = y1
                y1 += 30  # Augmenter y1 pour le prochain joueur

            # Dessiner le jeton de couleur
            self.canvas.create_oval(x1, y - radius, x2, y + radius, fill=color, outline=color)

            # Dessiner le texte
            self.canvas.create_text(x2 + 20, y, text=f"Player {i+1}", anchor="w")

    def draw_tracks(self):
        # Vérifiez si le jeu est terminé ou si la fenêtre principale n'existe plus
        if not self.is_running or not self.master.winfo_exists():
            return  # Sortir de la fonction si l'application n'est plus en cours d'exécution ou si la fenêtre principale n'existe plus
        track_lengths = {2: 3, 3: 5, 4: 7, 5: 9, 6: 11, 7: 13, 8: 11, 9: 9, 10: 7, 11: 5, 12: 3}
        track_width = 40
        start_x = 50
        start_y = 50
        space_between_tracks = 10

        for track_num, length in track_lengths.items():
            x = start_x + (track_width + space_between_tracks) * (track_num - 2)
            y = start_y
            for i in range(length):
                self.canvas.create_rectangle(x, y, x + track_width, y + 20, fill='lightgray', outline='black')
                self.canvas.create_text(x + track_width / 2, y + 10, text=str(track_num))
                y += 22  # Adding some space between rectangles

    def process_moves(self, current_player, dice_pairs):
        # Vérifiez si le jeu est terminé ou si la fenêtre principale n'existe plus
        if not self.is_running or not self.master.winfo_exists():
            return  # Sortir de la fonction si l'application n'est plus en cours d'exécution ou si la fenêtre principale n'existe plus
        self.clear_move_buttons()
        available_moves = {}
        move_number = 1

        for i, pair1 in enumerate(dice_pairs):
            for j, pair2 in enumerate(dice_pairs):
                if i != j:
                    sum1 = sum(pair1)
                    sum2 = sum(pair2)
                    track_pair = tuple(sorted((sum1, sum2)))

                    # Vérifier si aucune des pistes n'est gagnée
                    if not (track_pair[0] in self.won_columns or track_pair[1] in self.won_columns):
                        # Vérifier que le nombre de pistes actives n'est pas supérieur à 3
                        if len(self.active_tracks) < 3 or (sum1 in self.active_tracks and sum2 in self.active_tracks):
                            move_key = f"{move_number}. Move: {sum1} and {sum2} -> Tracks {track_pair[0]} and {track_pair[1]}"
                            available_moves[track_pair] = move_key
                            move_number += 1

        self.display_move_buttons(available_moves, current_player)

        #if not available_moves and len(self.active_tracks) >= 3:
        #    self.lose_markers(current_player)
    
    def get_available_moves(self, dice_pairs):
        available_moves = {}
        # Générer toutes les combinaisons possibles de paires de dés pour créer des mouvements
        for i, pair1 in enumerate(dice_pairs):
            for j, pair2 in enumerate(dice_pairs):
                if i != j:  # S'assurer que les paires de dés sont différentes pour former un mouvement
                    sum1 = sum(pair1)
                    sum2 = sum(pair2)
                    track_pair = tuple(sorted((sum1, sum2)))

                    # Vérifier si aucun des chemins n'est déjà gagné
                    if not (track_pair[0] in self.won_columns or track_pair[1] in self.won_columns):
                        # Vérifier si le nombre de pistes actives ne dépasse pas 3 ou si les pistes sont déjà actives
                        if len(self.active_tracks) < 3 or (track_pair[0] in self.active_tracks and track_pair[1] in self.active_tracks):
                            # Utiliser le tuple des sommes comme clé pour les mouvements disponibles
                            available_moves[track_pair] = (sum1, sum2)
        return available_moves
    
    def process_auto_player_move(self, move):
        sum1, sum2 = move
        # Vérifier si le mouvement est valide (p.ex. les pistes ne sont pas déjà gagnées)
        if sum1 not in self.won_columns and sum2 not in self.won_columns:
            # Mettre à jour les positions des jetons pour chaque somme du mouvement
            for sum in (sum1, sum2):
                # Si le joueur n'a pas encore de jeton sur cette piste, initialiser la position à 1
                print("self.player_track_positions[self.auto_player.name][sum] : ", self.player_track_positions[self.auto_player.name][sum])
                if self.player_track_positions[self.auto_player.name][sum] == 0:
                    self.player_track_positions[self.auto_player.name][sum] = 1
                else:
                    # Sinon, incrémenter la position du jeton
                    self.player_track_positions[self.auto_player.name][sum] += 1

                player_color = self.player_colors[self.game.players.index(self.auto_player)]
                # Mettre à jour visuellement le marqueur sur la piste
                self.update_track_markers(self.auto_player, sum, player_color)

            # Vérifier après chaque mouvement si cela mène à une victoire
            self.check_victory(self.auto_player)

    def is_move_valid(self, sum1, sum2):
        # Vérifier si les pistes ne sont pas déjà gagnées
        if sum1 in self.won_columns or sum2 in self.won_columns:
            return False

        # Vérifier si le joueur a déjà des marqueurs sur ces pistes ou si le nombre de pistes actives ne dépasse pas 3
        active_tracks = set(self.player_track_positions[self.auto_player.name].keys())
        if sum1 in active_tracks or sum2 in active_tracks:
            return True
        elif len(active_tracks) < 3:
            return True

        return False
    
    def update_game_state(self, sum1, sum2):
        # Mettre à jour les positions des marqueurs pour les pistes choisies
        self.player_track_positions[self.auto_player.name][sum1] += 1
        self.player_track_positions[self.auto_player.name][sum2] += 1

        # Vérifier si une colonne est gagnée
        if self.player_track_positions[self.auto_player.name][sum1] >= self.track_lengths[sum1]:
            self.won_columns[sum1] = self.auto_player.name
            self.update_won_column_display(sum1, self.auto_player.name)

        if self.player_track_positions[self.auto_player.name][sum2] >= self.track_lengths[sum2]:
            self.won_columns[sum2] = self.auto_player.name
            self.update_won_column_display(sum2, self.auto_player.name)

        # Vérifier si le joueur a gagné
        self.check_victory(self.auto_player)

    def update_won_column_display(self, track, player_name):
        track_width = 40
        start_x = 50
        space_between_tracks = 10
        x = start_x + (track_width + space_between_tracks) * (track - 2)

        # Utiliser la hauteur actuelle pour la position y du texte, puis l'incrémenter pour la prochaine utilisation
        y = self.track_text_heights[track]
        self.canvas.create_text(x + track_width / 2, y, text=f"{player_name} wins track {track}", fill="black")

        # Incrémenter la hauteur du texte pour cette colonne pour éviter la superposition avec le prochain texte ajouté
        self.track_text_heights[track] += 15  # Ajouter 15 pixels à la hauteur pour chaque nouveau texte


    def roll_dice(self):
        # Vérifiez si le jeu est terminé ou si la fenêtre principale n'existe plus
        if not self.is_running :
            return  # Sortir de la fonction si l'application n'est plus en cours d'exécution ou si la fenêtre principale n'existe plus
        
        self.chosen_move = None
        self.clear_move_buttons()

        # Récupérer l'objet Player pour le joueur actuel
        current_player = self.game.players[self.game.current_player_index]
        self.state_vector['game_phase'] = 0  # Mettre à jour la phase pour "choisir de s'arrêter ou continuer"

        # Si le joueur actuel est le joueur automatique, simuler le lancer de dés
        print("self.game.current_player_index lors du dice_roll = ", self.game.current_player_index)
        if current_player == self.auto_player:
            print(f"{current_player.name} is an auto player, simulating turn")
            #self.continue_playing_auto = True
            self.simulate_auto_player_turn()
        else : 
            #self.continue_playing_auto = False
            dice_pairs = current_player.roll_dice()
            self.update_action_vector(dice_pairs)
            result_text = f"Dice pairs: {', '.join([f'({d1}, {d2})' for d1, d2 in dice_pairs])}"
            self.process_moves(current_player, dice_pairs)
            self.dice_result_label.config(text=result_text)
            

    def clear_move_buttons(self):
        # Effacer les boutons de mouvement existants
        for btn in self.move_buttons:
            btn.destroy()
        self.move_buttons.clear()
    
    def initialize_game(self):
        # Choisir un joueur automatique au hasard
        self.auto_player = random.choice(self.game.players)
        print("Auto player is : ", self.auto_player.name)
        self.roll_dice()

    def simulate_auto_player_turn(self):
        print(f"Starting simulate_auto_player_turn for {self.auto_player.name}")
        if not self.is_running or not self.master.winfo_exists():
            print("simulate_auto_player_turn: Game is not running or master window does not exist")
            return
        self.continue_playing_auto = True # On assume qu'initialement le joueur veut jouer 
        # Tant que le joueur automatique décide de continuer
        while self.continue_playing_auto:
            dice_pairs = self.auto_player.roll_dice()
            print(f"Auto player {self.auto_player.name}'s turn with dice pairs: {dice_pairs}")

            available_moves = self.get_available_moves(dice_pairs)
            while not available_moves:
                print("No available moves for auto player. Relaunching dice.")
                dice_pairs = self.auto_player.roll_dice()
                print(f"Auto player {self.auto_player.name}'s turn with dice pairs: {dice_pairs}")
                available_moves = self.get_available_moves(dice_pairs)

            chosen_move = random.choice(list(available_moves.keys()))
            print(f"Auto player chooses to move on tracks {chosen_move}")
            lose_m = self.place_markers(chosen_move)
            
            if not lose_m :
                # Décision aléatoire pour continuer ou arrêter 
                self.continue_playing_auto = random.choice([True, False])
                print(f"Auto player {self.auto_player.name} {'decides to continue' if self.continue_playing_auto else 'decides to stop'}")
                if not self.continue_playing_auto:
                    self.continue_playing_auto = False
                    break
      
        # Si le joueur décide de s'arrêter ou s'il n'y a pas de mouvements disponibles
        if not self.continue_playing_auto:
            self.continue_playing_auto = False
            self.stop_game()

    def process_auto_player_move(self, move):
        if not self.is_running or not self.master.winfo_exists():
            return  # Sortir de la fonction si l'application n'est plus en cours d'exécution ou si la fenêtre principale n'existe plus
        print(f"Processing auto player move: {move}")
        self.place_bonzes(move, self.auto_player)
        self.check_victory(self.auto_player)

            
    def display_move_buttons(self, available_moves, current_player):
        # Vérifiez si la fenêtre principale existe encore avant d'interagir avec elle
        if not self.is_running :
            return  # Sortir de la fonction si l'application n'est plus en cours d'exécution ou si la fenêtre principale n'existe plus
        self.move_buttons.clear()
        self.clear_move_buttons()

        for widget in self.master.winfo_children():
            if isinstance(widget, tk.Button) and widget.cget("text").startswith("Move"):
                widget.destroy()
        
        for move_num, move_desc in available_moves.items():
            btn = tk.Button(self.master, text=move_desc)
            btn.config(command=lambda mn=move_num, b=btn: self.choose_move(mn, b))
            btn.pack()
            self.move_buttons.append(btn)
    
    def reset_temporary_markers(self):
        for marker_id in self.current_turn_markers:
            self.canvas.delete(marker_id)
        self.current_turn_markers.clear()

    def choose_move(self, move_num, btn):
        # Réinitialiser les marqueurs temporaires du tour actuel
        self.reset_temporary_markers()
        
        # Réinitialiser tous les boutons sauf le bouton actuellement choisi
        for button in self.move_buttons:
            if button != btn:
                button.config(state="normal", bg="white")

        # Griser le bouton choisi et stocker le mouvement choisi
        btn.config(state="disabled", bg="lightgrey")
        self.chosen_move = move_num
        self.chosen_move_label.config(text=f"Chosen move: {self.chosen_move}")

        # Placer de nouveaux marqueurs pour les pistes choisies
        self.place_markers(move_num)

        # Supprimer le choix précédent s'il existe
        self.chosen_move = None
        self.chosen_move_label.config(text="")

        # Effacer les boutons de mouvement existants
        self.clear_move_buttons()
    
    def clear_current_turn_markers(self):
        # Supprimer les marqueurs du tour actuel
        for marker_id in self.current_turn_markers:
            self.canvas.delete(marker_id)
        self.current_turn_markers.clear()

    def initialize_player_markers(self):
        for player_name, markers in self.player_markers.items():
            player_color = self.player_colors[self.game.players.index(self.game.get_player_by_name(player_name))]
            for track_num, position in self.player_track_positions[player_name].items():
                x, y = self.calculate_marker_position(track_num, position)
                marker_id = self.canvas.create_oval(x, y, x + 10, y + 10, fill=player_color)
                markers.append(marker_id)

    def calculate_marker_position(self, track_num, position):
        track_width = 40
        start_x = 50
        space_between_tracks = 10
        x = start_x + (track_width + space_between_tracks) * (track_num - 2)
        y = 50 + 20 * position
        return x, y

    def update_track_markers(self, player, track_num, color):
        if not self.is_running:
            return  # S'arrêter si le canvas n'existe plus
        x, y = self.calculate_marker_position(track_num, self.player_track_positions[player.name][track_num] - 1)

        # Supprimer le marqueur précédent de cette colonne, s'il existe
        marker_index = track_num - 2
        if self.player_markers[player.name][marker_index] is not None:
            if self.canvas.winfo_exists():
                self.canvas.delete(self.player_markers[player.name][marker_index])

        # Créer un nouveau marqueur pour cette colonne
        marker_id = self.canvas.create_oval(x, y, x + 10, y + 10, fill=color)
        self.player_markers[player.name][marker_index] = marker_id

    def place_markers(self, move_num):
        lose_m = False
        if not self.is_running:
            return  # S'arrêter si le canvas n'existe plus
        current_player = self.game.players[self.game.current_player_index]
        player_color = self.player_colors[self.game.current_player_index]

        print(f"Current move: {move_num}")
        print(f"Active tracks before move: {self.active_tracks}")

        # Utiliser un ensemble pour garder une trace des pistes activées par ce mouvement
        tracks_this_turn = set(move_num)

        # Calculer l'ensemble des pistes actives après ce mouvement sans encore les ajouter
        potential_active_tracks = self.active_tracks.union(tracks_this_turn)

        print(f"Tracks activated this turn: {tracks_this_turn}")
        print(f"New tracks activated: {tracks_this_turn.difference(self.active_tracks)}")
        print(f"Active tracks after move (potentially): {potential_active_tracks}")

        if len(potential_active_tracks) > 3:
            print("Too many active tracks - losing markers!")
            self.lose_markers(current_player)
            lose_m = True 
            
        else:
            # Seulement si le joueur ne perd pas ses marqueurs, mettre à jour les pistes actives
            self.active_tracks = potential_active_tracks

            for track_num in move_num:
                # Ajouter ou mettre à jour les progrès sur cette piste
                self.player_track_positions[current_player.name][track_num] += 1

                # Mettre à jour le marqueur sur la piste
                self.update_track_markers(current_player, track_num, player_color)

            # Si le mouvement est valide, demander au joueur s'il veut continuer ou arrêter
            self.ask_continue_or_stop()
        return lose_m


    def ask_continue_or_stop(self):
        if not self.is_running:
            return  # S'arrêter si le canvas n'existe plus
        # Effacer la frame existante si elle existe
        if hasattr(self, 'decision_frame'):
            if self.decision_frame is not None:
                self.decision_frame.destroy()
                self.decision_frame = None  # Réinitialiser decision_frame à None après la destruction


        # Créer une nouvelle frame pour les boutons de décision
        self.decision_frame = tk.Frame(self.master)
        self.decision_frame.pack()

        # Créer les boutons "Continuer" et "Arrêter"
        continue_button = tk.Button(self.decision_frame, text="Continuer", command=self.continue_game)
        continue_button.pack(side=tk.LEFT)

        stop_button = tk.Button(self.decision_frame, text="Arrêter", command=self.stop_game)
        stop_button.pack(side=tk.RIGHT)

    def continue_game(self):
        if not self.is_running:
            return  # S'arrêter si le canvas n'existe plus
        # Vérifie si decision_frame existe avant de le détruire
        if self.decision_frame is not None:
            self.decision_frame.destroy()
            self.decision_frame = None
        self.roll_dice()
        
    def lose_markers(self, current_player):
        if not self.is_running:
            return  # S'arrêter si le canvas n'existe plus
        player_name = current_player.name
        print(f"{player_name} perd ses marqueurs !")
        messagebox.showinfo("Perte", f"{player_name} perd ses marqueurs !")
        
        if current_player == self.auto_player :
                self.continue_playing_auto = False

        for track_num in self.active_tracks:
            # Réinitialiser les marqueurs à leur dernière position permanente
            last_valid_position = self.previous_permanent_positions[player_name][track_num]
            self.player_track_positions[player_name][track_num] = last_valid_position

            # Mettre à jour visuellement le marqueur
            self.update_track_markers_from_lose_markers(current_player, track_num)

        self.active_tracks.clear()
        if current_player != self.auto_player : 
            self.next_player_or_end_game()
        
    def update_track_markers_from_lose_markers(self, player, track_num):
        position = self.player_track_positions[player.name][track_num]
        if position > 0:  # Assurez-vous qu'il y a un progrès à visualiser
            x, y = self.calculate_marker_position(track_num, position)
            marker_index = track_num - 2
            # Supprimez le marqueur précédent, s'il existe
            if self.player_markers[player.name][marker_index] is not None:
                self.canvas.delete(self.player_markers[player.name][marker_index])
            # Créez un nouveau marqueur à la position mise à jour
            marker_id = self.canvas.create_oval(x - 5, y - 10, x + 5, y, fill=self.player_colors[self.game.players.index(player)])
            self.player_markers[player.name][marker_index] = marker_id
        else:
            # S'il n'y a pas de progrès (position == 0), assurez-vous qu'aucun marqueur n'est affiché
            marker_index = track_num - 2
            if self.player_markers[player.name][marker_index] is not None:
                self.canvas.delete(self.player_markers[player.name][marker_index])
                self.player_markers[player.name][marker_index] = None
        
    def recalculate_permanent_markers(self):
        if not self.canvas.winfo_exists():
            return  # Ne rien faire si le canvas n'existe plus
        print("Recalcul des marqueurs permanents pour tous les joueurs")
        for player in self.game.players:
            player_name = player.name
            for track_num, position in self.player_track_positions[player_name].items():
                if position > 0:  # Si le joueur a un marqueur permanent sur cette voie
                    print(f"Mise à jour du marqueur permanent pour {player_name} sur la voie {track_num}")
                    self.update_track_markers(player, track_num, self.player_colors[self.game.players.index(player)])
                    
    def stop_game(self):
        if not self.canvas.winfo_exists():
            return  # Ne rien faire si le canvas n'existe plus
        print("Arrêt du tour. Conversion des marqueurs temporaires en permanents.")
        # Convertir les marqueurs temporaires en permanents
        self.convert_temporary_to_permanent_markers()

        current_player = self.game.players[self.game.current_player_index]
        print(f"Fin du tour pour {current_player.name}. Vérification de la victoire et passage au joueur suivant.")

        self.check_victory(current_player)

        if self.decision_frame is not None:
            self.decision_frame.destroy()
            self.decision_frame = None

        self.active_tracks.clear()
        self.next_player_or_end_game()
        self.recalculate_permanent_markers()
        
    def convert_temporary_to_permanent_markers(self):
        if not self.canvas.winfo_exists():
            return  # Ne rien faire si le canvas n'existe plus
        current_player = self.game.players[self.game.current_player_index]
        player_name = current_player.name

        for track_num in self.active_tracks:
            position = self.player_track_positions[player_name][track_num]
            # Assurez-vous que la position est supérieure à 0 avant de considérer comme permanente
            if position > 0:
                # Mettre à jour le marqueur permanent
                self.previous_permanent_positions[player_name][track_num] = position

        # Réinitialiser les marqueurs temporaires pour le tour suivant
        self.current_turn_markers.clear()


    def next_player_or_end_game(self):
        # Vérifiez si le jeu est terminé ou si la fenêtre principale n'existe plus
        if not self.is_running :
            return  # Sortir de la fonction si l'application n'est plus en cours d'exécution ou si la fenêtre principale n'existe plus

        if self.game.is_game_over():
            # Marquer le jeu comme terminé et fermer la fenêtre
            self.game_over = True
            self.is_running = False
            messagebox.showinfo("Fin du jeu", "Le jeu est terminé !")
            if self.master.winfo_exists():
                self.master.destroy()
            return
        
        # Enregistrez les positions précédentes des marqueurs permanents pour le joueur actuel
        current_player = self.game.players[self.game.current_player_index]
        previous_permanent_positions = self.previous_permanent_positions[current_player.name].copy()

        # Réinitialisez les marqueurs temporaires pour le tour suivant en utilisant les positions précédentes
        for track_num in self.active_tracks:
            if self.player_track_positions[current_player.name][track_num] == 0:
                # Si la position actuelle est 0 (aucun marqueur permanent), rétablissez la position précédente
                self.player_track_positions[current_player.name][track_num] = previous_permanent_positions[track_num]
        # Passer au joueur suivant
        
        self.game.current_player_index = (self.game.current_player_index + 1) % self.num_players
        print(f"It's now Player {self.game.current_player_index + 1}'s turn.")
        self.active_tracks.clear()
        
        self.roll_dice()

    def check_victory(self, current_player):
        if not self.canvas.winfo_exists():
            return  # Ne rien faire si le canvas n'existe plus
        # Compter le nombre de colonnes gagnées par le joueur actuel
        columns_won_by_current_player = list(self.won_columns.values()).count(current_player.name)

        # Vérifier si le joueur actuel a gagné 3 colonnes
        if columns_won_by_current_player >= 3:
            messagebox.showinfo("Victoire", f"{current_player.name} a gagné la partie !")
            self.game_over = True
            if self.master.winfo_exists():
                self.master.destroy()
            return True

        # Vérification existante pour chaque piste, si nécessaire
        for track, position in self.player_track_positions[current_player.name].items():
            if position >= self.track_lengths[track]:  # Vérification de la colonne gagnée
                if track not in self.won_columns:
                    self.won_columns[track] = current_player.name
                    self.update_won_column_display(track, current_player.name)

        # La partie n'est pas encore gagnée
        return False
    
    def on_close(self):
        self.is_running = False
        self.master.destroy()
