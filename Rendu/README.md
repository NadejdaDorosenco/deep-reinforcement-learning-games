# Infos générales

- Les images d'expérimentations se trouvent dans le dossier Images expérimentations/


# Pour can't stop, GridWorld, LineWorld, TicTacToe : 

## Exécution du main :

Exécution du main.py qui simule un jeu joueur random vs 1 ou plusieurs joueurs humains: 
	- choix du mode (terminal T ou GUI G)
	- choix du nombre de joueurs : 2 à 4

## Exécution des modèles :

Les modèles DQN, DDQN, DDQNWER, DDQNWPER et Random Rollout peuvent être testés sur les 4 environnements GridWorld, LineWorld, TicTacToe et Can't Stop. Tous ces modèles sont dans le sous dossier cantstop.

Chaque modèle est dans un fichier, pour les exécuter : python3 nomdumodèle.py.
Pour changer d'environnement d'éxécution pour un modèle, il faut changer l'initialisation de l'agent et l'action_size et l'OBS dans les paramètres l'agent lors de son initialisation. Il faut juste commenter celui qui est dans le fichier (initialement environnement can't stop) et décommenter les lignes correspondant à l'environnement voulu.

Concernant le modèle MCTS, nous ne sommes pas arrivés à l'exécuter par défaut de mémoire à notre avis. 

PPO s'exécute avec TicTacToe par le fichier balloonpop/PPO.py donc : python3 balloonpop/PPO.py 


PPO s'exécute avec CantStop par le fichier cantstop/PPO.py donc : python3 cantstop/PPO.py 


# Pour balloon pop: 

Compiler et exécuter main.py puis préciser l'algorithme qui nous intéresse.

Solo correspond au jeu GUI.
Les autres modes de jeu sont : random, DQN et DDQN.

Les fichiers sont divisés en model, view controller.
view.py et controller.py ne sont utilisés que par le jeu "solo".

DeepEnv, sert à récupérer les éléments essentiels de model qui sont réutilisés dans DQN et DDQN.

