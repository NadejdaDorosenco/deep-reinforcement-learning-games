import pygame
from model import BalloonGameSolo
from view import PygameView
import time


class GameController:
    
    def __init__(self, model, view):
        self.model = model
        self.model.roll_dice()
        self.model.record_actual_score()
        self.view = view
        self.running = True
    
    def check_for_breaks(self):
        break_occurred = self.model.check_for_breaks()
        if break_occurred:
            print(f"Break dans une colonne !")

    def update_positions(self, num_dice):
        self.target_positions = [(self.view.plateau_pos[0] + self.view.plateau_image.get_width() + 70, 
                              self.view.plateau_pos[1] + 40 + i * 60) for i in range(num_dice)]
        # Initialisez les positions pour les dés supplémentaires en dehors de l'écran
        for i in range(num_dice, len(self.view.dice_images)):
            self.target_positions.append((self.view.WINDOW_SIZE[0] + 100, self.view.WINDOW_SIZE[1] + 100))
        self.dice_positions = list(self.target_positions)
    
    def all_dices_kept(self):
        return all([pos[0] >= self.view.conserver_position for pos in self.view.dice_positions[:5]])
    
    def handle_dice_clicked(self, dice_to_reroll):
        dice_to_toggle = dice_to_reroll
        if self.view.dice_positions[dice_to_toggle][0] == self.view.target_positions[dice_to_toggle][0] == self.view.conserver_position:
            self.view.target_positions[dice_to_toggle] = (self.view.plateau_pos[0] + self.view.plateau_image.get_width() + 70, 
                                                        self.view.plateau_pos[1] + 40 + dice_to_toggle * 60)
        else:
            self.view.target_positions[dice_to_toggle] = (self.view.conserver_position, self.view.dice_positions[dice_to_toggle][1])
        
    def update_screen(self):
        self.view.display(self.model) 
        pygame.display.update()

    def handle_mouse_released(self):
        if self.view.selected_dice is not None:
            self.view.dice_positions[self.view.selected_dice] = self.view.conserver_position
            self.view.selected_dice = None
        
    def handle_dice_not_kept(self, dice_to_reroll):
        dice_to_reroll = [i for i in dice_to_reroll if i < len(self.model.dice_set)]
        previous_dice_count = len(self.model.dice_set)
        self.model.roll_dice(dice_to_reroll)
        self.view.update_dice_positions(len(self.model.dice_set))
        self.update_positions(len(self.model.dice_set))
        # Mettre à jour les positions des dés relancés
        for i in range(len(self.model.dice_set)):
            # Si le dé était précédemment sous "relancer" ou s'il s'agit d'un nouveau dé ajouté lors de la relance
            if i in dice_to_reroll or i >= previous_dice_count:
                self.view.dice_positions[i] = (self.view.plateau_pos[0] + self.view.plateau_image.get_width() + 70, 
                                            self.view.plateau_pos[1] + 40 + i * 60)
                self.view.target_positions[i] = self.view.dice_positions[i]
            else:
                # Si le dé était précédemment sous "conserver"
                self.view.dice_positions[i] = (self.view.conserver_position, self.view.plateau_pos[1] + 40 + i * 60)
                self.view.target_positions[i] = self.view.dice_positions[i]

    def reinit_dices(self):
        self.model.dice_set = self.model.original_dice_set.copy()
        self.update_positions(len(self.model.dice_set))
        self.model.record_scores()
        self.model.rolls = []
        self.model.roll_dice()
        # Réinitialise les positions des dés dans la vue
        self.view.dice_positions = [(self.view.plateau_pos[0] + self.view.plateau_image.get_width() + 70, 
                                    self.view.plateau_pos[1] + 40 + i * 60) for i in range(len(self.model.dice_set))]
        self.view.target_positions = list(self.view.dice_positions)  # Réinitialise les positions cibles
        self.view.reroll_count = 0  # Réinitialise le compteur de relance pour le prochain tour
        self.check_for_breaks()
    
    def handle_reroll_clicked(self, dice_to_reroll):
        if self.all_dices_kept():
            return  # Do nothing if all dices are kept
        all_dices_kept = all([pos[0] >= self.view.conserver_position for pos in self.view.dice_positions[:5]])
        if not all_dices_kept:  # Ajoutez cette condition
            self.handle_dice_not_kept(dice_to_reroll)
        # Si le compteur de relance est à 2, on affiche les dés, on attend 2 secondes puis on passe au tour suivant
        if self.view.reroll_count == 2:
            self.handle_conserver_tout()

    def handle_conserver_tout(self):
        self.update_screen()
        time.sleep(2)
        self.reinit_dices()        

    def run(self):
        clock = pygame.time.Clock()
        FPS = 60
        while self.running:
            dice_to_reroll = []  # Initialisation par défaut
            event, dice_to_reroll = self.view.get_input()
            self.model.record_actual_score()
            if event == "dice_clicked":
                self.handle_dice_clicked(dice_to_reroll)
            elif event == "mouse_released":
                self.handle_mouse_released()
            elif event == "reroll_clicked":
                self.handle_reroll_clicked(dice_to_reroll)
            elif event == "conserver_tout_clicked":
                self.handle_conserver_tout()
            elif event == "quit":
                self.running = False

            self.update_screen()
            clock.tick(FPS)

            # Si le troisième break a été atteint, affichez la fin du jeu
            if self.model.break_count >= 3:
                time.sleep(10)  # Pause pendant 10 secondes
                self.running = False  # Arrêtez le jeu après la fin du troisième break
