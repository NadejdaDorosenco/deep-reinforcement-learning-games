import pygame
from pygame.locals import QUIT, MOUSEBUTTONDOWN, MOUSEBUTTONUP
import sys

class PygameView:
    def __init__(self, model):
        self.model = model
        pygame.init()

        # paramêtre windows
        self.WINDOW_SIZE = (1000, 800)
        self.window = pygame.display.set_mode(self.WINDOW_SIZE, pygame.RESIZABLE)
        pygame.display.set_caption('Jeu de Ballon')
        
        # element visuelle
        self._init_resources()
        self._init_positions()
        self._init_buttons()
        self._init_dice()
    
    def display_score_final(self, score):
        break_score_font = pygame.font.SysFont(None, 36)
        # Efface la zone du score final
        background_color = (255, 255, 255)
        final_score_size = break_score_font.size(str(40))
        final_score_pos = (440, 285)
        pygame.draw.rect(self.window, background_color, final_score_pos + final_score_size)
        # affiche le nouveau score final
        text_final = str(score)
        final_score = break_score_font.render(text_final, True, (190,24,69))
        self.window.blit(final_score, (440, 285))

    def display_break_scores(self):
        break_score_font = pygame.font.SysFont(None, 36)
        if len(self.model.break_scores) > 0:
            # affichage score 1er break
            text = str(self.model.break_scores[0])
            break_score_text = break_score_font.render(text, True, (48,183,0))
            self.window.blit(break_score_text, (80, 200))
            # affichage score final
            final_score = break_score_font.render(text, True, (190,24,69))
            self.window.blit(final_score, (440, 285))
        if len(self.model.break_scores) > 1:
            # affichage score 2eme break
            text_break = str(self.model.break_scores[1])
            break_score_text = break_score_font.render(text_break, True, (224, 195, 70))
            self.window.blit(break_score_text, (80, 278))
            # Efface la zone du score final
            self.display_score_final(self.model.break_scores[0] + self.model.break_scores[1])
        if len(self.model.break_scores) > 2:
            # affichage score 3eme break
            text = str(self.model.break_scores[2])
            break_score_text = break_score_font.render(text, True, (220, 124, 0))
            self.window.blit(break_score_text, (440, 200))
            # Efface la zone du score final
            self.display_score_final(self.model.break_scores[0] + self.model.break_scores[1] + self.model.break_scores[2])

    def _init_resources(self):
        self.background_image_original = pygame.image.load('Ressources/fond.jpg')
        self.background_image = pygame.transform.scale(self.background_image_original, self.WINDOW_SIZE)
        self.font = pygame.font.Font(None, 24)
        self.scaled_size = (45, 45)
        self.plateau_image_original = pygame.image.load('Ressources/plateau.png')

    def _init_positions(self):
        plateau_width = int(0.45 * self.WINDOW_SIZE[0])
        scale_factor = plateau_width / self.plateau_image_original.get_width()
        plateau_height = int(scale_factor * self.plateau_image_original.get_height())
        self.plateau_image = pygame.transform.scale(self.plateau_image_original, (plateau_width, plateau_height))
        self.plateau_margin = 50
        self.plateau_pos = (self.plateau_margin, self.WINDOW_SIZE[1] - self.plateau_image.get_height() - self.plateau_margin)

        # Positions for scoring
        column_spacing = self.plateau_image.get_width() / 6
        max_rows = len(self.model.columns["Ballon rouge"])
        row_spacing = self.plateau_image.get_height() / max_rows
        self.score_positions = {}
        for index, (column_name, scores) in enumerate(self.model.columns.items()):
            column_x = self.plateau_pos[0] + (index * column_spacing)
            positions = [(column_x, self.plateau_pos[1] + self.plateau_image.get_height() - (i * row_spacing))
                        for i in range(len(scores))]
            self.score_positions[column_name] = positions
            
        # General positions
        base_positions_y = [710, 668, 620, 575, 530, 485, 439, 392, 345, 298]
        object_names = ["Ballon jaune", "Ballon bleu", "Ballon rouge", "Etoile", "Lune", "Losange"]
        object_x_positions = [105, 180, 250, 320, 390, 450]
        positions_count = [6, 7, 10, 9, 8, 6]
        self.positions = self.generate_positions(object_names, object_x_positions, positions_count, base_positions_y)


    def _init_buttons(self):
        self.button_relancer = pygame.Rect(550, 15, 100, 30)
        self.button_conserver = pygame.Rect(660, 15, 180, 30)
        self.button_color = (84, 114, 174)
        self.shadow_color = (50, 50, 50)
        self.shadow_offset = 3
        self.border_radius = 5
    
    def _init_dice(self):
        # Charger les images de dés
        self.dice_images = {
            "Rouge etoile": pygame.transform.scale(pygame.image.load('Ressources/rouge_etoile.png'), self.scaled_size),
            "Rouge lune": pygame.transform.scale(pygame.image.load('Ressources/rouge_lune.png'), self.scaled_size),
            "Rouge losange": pygame.transform.scale(pygame.image.load('Ressources/rouge_losange.png'), self.scaled_size),
            "Bleu etoile": pygame.transform.scale(pygame.image.load('Ressources/bleu_etoile.png'), self.scaled_size),
            "Bleu lune": pygame.transform.scale(pygame.image.load('Ressources/bleu_lune.png'), self.scaled_size),
            "Jaune etoile": pygame.transform.scale(pygame.image.load('Ressources/jaune_etoile.png'), self.scaled_size),
        }
        self.selected_dice = None
        self.dice_positions = [(self.plateau_pos[0] + self.plateau_image.get_width() + 70, 
                                self.plateau_pos[1] + 40 + i * 60) for i in range(len(self.dice_images))]
        self.target_positions = list(self.dice_positions)
        self.conserver_position = self.plateau_pos[0] + self.plateau_image.get_width() + 220
        self.move_speed = 4
        self.reroll_count = 0

    def generate_positions(self, object_names, object_x_positions, positions_count, base_positions_y):
        positions = {}
        for obj_name, x, count in zip(object_names, object_x_positions, positions_count):
            positions[obj_name] = [(x, y) for y in base_positions_y[:count]]
        return positions
     
    def display(self, game):
        self.window.blit(self.background_image, (0, 0))
        self._draw_banderole()
        self._display_text_and_buttons()
        self._display_buttons()
        self._display_dice(game)
        self.window.blit(self.plateau_image, self.plateau_pos)
        self.draw_scores()  
        if self.model.break_count < 3:
            self.draw_actual_scores()  
        self.display_break_scores()  
    
    def draw_scores(self):
        for column_name, positions in self.positions.items():
            # Obtenir la progression actuelle pour cette colonne
            progression = self.model.progression[column_name]
            # Si la progression est au moins de 0 (c'est-à-dire que le joueur a marqué des points dans cette colonne)
            if progression >= 0:
                # Obtenir la position centrale de la case correspondante
                center_x, center_y = positions[progression]
                # Définir un rayon pour le cercle. 
                radius = 25 
                pygame.draw.circle(self.window, (255, 0, 0), (center_x, center_y), radius, 2)  # 2 est l'épaisseur de la ligne

    def draw_actual_scores(self):
        for column_name, positions in self.positions.items():
            # Obtenir la progression actuelle pour cette colonne
            actual_progression = self.model.actual_progression[column_name]
            recorded_progression = self.model.progression[column_name]

            # Si la progression est au moins de 0 et que actual_progression != recorded_progression
            if actual_progression >= 0 and actual_progression != recorded_progression:
                # Obtenir la position centrale de la case correspondante
                center_x, center_y = positions[actual_progression]
                radius = 25 
                temp_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surface, (255, 0, 0, 75), (radius, radius), radius)
                self.window.blit(temp_surface, (center_x - radius, center_y - radius))
                pygame.draw.circle(self.window, (255, 0, 0), (center_x, center_y), radius, 2)  # 2 est l'épaisseur de la ligne
    
    def draw_circle_around_score(self, column, index):
        """Draws a circle around the specified score."""
        position = self.score_positions[column][index]
        pygame.draw.circle(self.window, (255, 0, 0), position, 20, 2)

    def get_input(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEBUTTONDOWN:
                x, y = event.pos
                # Vérification des clics sur les dés
                for i, result in enumerate(self.model.rolls):
                    dice_rect = pygame.Rect(*self.dice_positions[i], *self.scaled_size)
                    if dice_rect.collidepoint(x, y):
                        return "dice_clicked", i
                # Vérification du clic sur le bouton "Relancer !"
                if self.button_relancer.collidepoint(x, y) and self.reroll_count < 2:
                    dice_to_reroll = [i for i, pos in enumerate(self.dice_positions[:5]) if pos[0] < self.conserver_position]
                    self.reroll_count += 1
                    return "reroll_clicked", dice_to_reroll
                 # Vérification du clic sur le bouton "Tous les conserver !"
                if self.button_conserver.collidepoint(x, y):
                    return "conserver_tout_clicked", []
            elif event.type == MOUSEBUTTONUP:
                if self.selected_dice is not None:
                    return "mouse_released", []
            elif event.type == pygame.VIDEORESIZE:
                self.WINDOW_SIZE = (event.w, event.h)
                self.window = pygame.display.set_mode(self.WINDOW_SIZE, pygame.RESIZABLE)
                self.background_image = pygame.transform.scale(self.background_image_original, self.WINDOW_SIZE)
                return "resize", []
        return None, []

    def update_dice_positions(self, num_dice):
        self.dice_positions = [(self.plateau_pos[0] + self.plateau_image.get_width() + 70, 
                                self.plateau_pos[1] + 40 + i * 60) for i in range(num_dice)]
        self.target_positions = list(self.dice_positions)

    def _draw_banderole(self):
        pygame.draw.rect(self.window, (255, 255, 255), (0, 10, self.WINDOW_SIZE[0], 40))

    def _display_text_and_buttons(self):
        # Texte "Vous lancez les dés"
        vous_text = self.font.render('Vous', True, (255, 0, 0))
        lancez_text = self.font.render('lancez les dés', True, (0, 0, 0))
        
        total_width = vous_text.get_width() + lancez_text.get_width() + 7  # 7 est l'espace entre les deux textes
        start_x = (self.WINDOW_SIZE[0] - total_width) / 2
        
        self.window.blit(vous_text, (start_x, 17))
        self.window.blit(lancez_text, (start_x + vous_text.get_width() + 7, 17))

        # Boutons et textes "Relancer" et "Conserver"
        relancer_text_position = (self.plateau_pos[0] + self.plateau_image.get_width() + 60, self.plateau_pos[1])
        self.window.blit(self.font.render('Relancer', True, (0, 0, 0)), relancer_text_position)
        
        conserver_text_position = (relancer_text_position[0] + 140, relancer_text_position[1])
        self.window.blit(self.font.render('Conserver', True, (0, 0, 0)), conserver_text_position)

        # Boutons (visuellement)
        self._display_buttons()


    def _display_buttons(self):
        shadow_relancer = self.button_relancer.move(self.shadow_offset, self.shadow_offset)
        shadow_conserver = self.button_conserver.move(self.shadow_offset, self.shadow_offset)
        
        pygame.draw.rect(self.window, self.shadow_color, shadow_relancer, border_radius=self.border_radius)
        pygame.draw.rect(self.window, self.shadow_color, shadow_conserver, border_radius=self.border_radius)
        pygame.draw.rect(self.window, self.button_color, self.button_relancer, border_radius=self.border_radius)
        pygame.draw.rect(self.window, self.button_color, self.button_conserver, border_radius=self.border_radius)
        
        relancer_text = self.font.render('Relancer !', True, (255, 255, 255))
        conserver_text = self.font.render('Tous les conserver !', True, (255, 255, 255))
        
        self.window.blit(relancer_text, (self.button_relancer.x + 10, self.button_relancer.y + 7))
        self.window.blit(conserver_text, (self.button_conserver.x + 10, self.button_conserver.y + 7))
        
        # Ajustement de la position des boutons pour éviter le chevauchement
        dynamic_spacing = 0.05 * self.WINDOW_SIZE[0]
        total_text_width = (self.WINDOW_SIZE[0] / 2) + (relancer_text.get_width() / 2)
        min_x_for_button = total_text_width + dynamic_spacing
        
        if self.button_relancer.x < min_x_for_button:
            self.button_relancer.x = min_x_for_button
        if self.button_conserver.x < min_x_for_button + self.button_relancer.width + 10:
            self.button_conserver.x = min_x_for_button + self.button_relancer.width + 10

    def _display_dice(self, game):
        # Limitez l'énumération à un maximum de 5 dés
        for i, result in enumerate(game.rolls[:5]):
            if result in self.dice_images:
                distance_to_target = self.target_positions[i][0] - self.dice_positions[i][0]
                if abs(distance_to_target) <= self.move_speed:
                    self.dice_positions[i] = self.target_positions[i]
                else:
                    if self.dice_positions[i][0] < self.target_positions[i][0]:
                        self.dice_positions[i] = (self.dice_positions[i][0] + self.move_speed, self.dice_positions[i][1])
                    elif self.dice_positions[i][0] > self.target_positions[i][0]:
                        self.dice_positions[i] = (self.dice_positions[i][0] - self.move_speed, self.dice_positions[i][1])

                self.window.blit(self.dice_images[result], self.dice_positions[i])
            else:
                print(f"Error: Image for result {result} not found!")
    
    def display_score(self, game):
        score_start_x = self.plateau_pos[0] + self.plateau_image.get_width() + 20  
        score_start_y = self.plateau_pos[1] + self.plateau_image.get_height() - 20  

        score_font = pygame.font.Font(None, 16)

        for index, (column_name, progression) in enumerate(game.progression.items()):
            score = game.columns[column_name][progression] if progression >= 0 else 0
            score_text = score_font.render(f"{column_name}: {score}", True, (0, 0, 0))

            score_position = (score_start_x, score_start_y - (index * 20)) 

            self.window.blit(score_text, score_position)

    def get_input(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEBUTTONDOWN:
                x, y = event.pos
                # Vérification des clics sur les dés
                for i, result in enumerate(self.model.rolls):
                    dice_rect = pygame.Rect(*self.dice_positions[i], *self.scaled_size)
                    if dice_rect.collidepoint(x, y):
                        return "dice_clicked", i
                # Vérification du clic sur le bouton "Relancer !"
                if self.button_relancer.collidepoint(x, y) and self.reroll_count < 2:
                    dice_to_reroll = [i for i, pos in enumerate(self.dice_positions[:5]) if pos[0] < self.conserver_position]
                    self.reroll_count += 1
                    return "reroll_clicked", dice_to_reroll
                 # Vérification du clic sur le bouton "Tous les conserver !"
                if self.button_conserver.collidepoint(x, y):
                    return "conserver_tout_clicked", []
            elif event.type == MOUSEBUTTONUP:
                if self.selected_dice is not None:
                    return "mouse_released", []
            elif event.type == pygame.VIDEORESIZE:
                self.WINDOW_SIZE = (event.w, event.h)
                self.window = pygame.display.set_mode(self.WINDOW_SIZE, pygame.RESIZABLE)
                self.background_image = pygame.transform.scale(self.background_image_original, self.WINDOW_SIZE)
                return "resize", []
        return None, []
