from game import Game
import tkinter as tk
from gamegui import GameGUI

if __name__ == "__main__":
    print("Welcome to Can't Stop")
    num_players = int(input("Enter the number of players between 2 and 4: "))
    if num_players < 2 or num_players > 4 :
        print("Wrong number of players ")
    else : 
        choice = input("GUI (G) or Terminal (T) ?").upper()
        if choice == "T":
            player_colors = ["red", "blue", "green", "yellow"][:num_players]
            game = Game(num_players,player_colors)
            game.start_game()
            print("Action Vector:", game.action_vector)
            print("State Vector:", game.state_vector)
        elif choice == "G":
            # Création de la fenêtre Tkinter
            root = tk.Tk()
            app = GameGUI(root, num_players)
            #app = GameGUI(root, num_players, game_mode='random-player')
            app.initialize_game()
            root.mainloop()
            print("Action Vector:", app.action_vector)
            print("State Vector:", app.state_vector)
        else :
            print("Choice not G or T")

