import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from timeit import default_timer
from ai import Dqn  # Import de notre réseau neuronal
import pygame

# Initialisation de Pygame
pygame.init()

# Définition des constantes
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
CAR_SIZE = 20  # Taille de la voiture
SENSOR_LENGTH = 30  # Longueur des capteurs
BUTTON_HEIGHT = 40
BUTTON_WIDTH = 100

class Car:
    def __init__(self, x, y):
        self.x = x  # Position X initiale
        self.y = y  # Position Y initiale
        self.angle = 0  # Angle de la voiture
        self.rotation = 0  # Rotation actuelle
        self.velocity = 6  # Vitesse scalaire unique (comme dans Kivy)
        # Initialisation des capteurs avec des tenseurs PyTorch
        self.sensor1 = torch.zeros(2)  # Capteur avant
        self.sensor2 = torch.zeros(2)  # Capteur droit
        self.sensor3 = torch.zeros(2)  # Capteur gauche
        self.signal1 = 0  # Signal du capteur avant
        self.signal2 = 0  # Signal du capteur droit
        self.signal3 = 0  # Signal du capteur gauche
        
    def move(self, rotation):
        # Mise à jour de la rotation et de l'angle
        self.rotation = rotation
        self.angle = (self.angle + self.rotation) % 360
        
        # Conversion de l'angle en radians
        angle_rad = self.angle * np.pi / 180
        
        # Mise à jour de la position selon l'angle
        self.x += self.velocity * np.cos(angle_rad)
        self.y += self.velocity * np.sin(angle_rad)
        
        # Mise à jour des positions des capteurs
        # Capteur avant
        self.sensor1[0] = self.x + SENSOR_LENGTH * np.cos(angle_rad)
        self.sensor1[1] = self.y + SENSOR_LENGTH * np.sin(angle_rad)
        
        # Capteur droit (+30 degrés)
        angle_rad2 = ((self.angle + 30) % 360) * np.pi / 180
        self.sensor2[0] = self.x + SENSOR_LENGTH * np.cos(angle_rad2)
        self.sensor2[1] = self.y + SENSOR_LENGTH * np.sin(angle_rad2)
        
        # Capteur gauche (-30 degrés)
        angle_rad3 = ((self.angle - 30) % 360) * np.pi / 180
        self.sensor3[0] = self.x + SENSOR_LENGTH * np.cos(angle_rad3)
        self.sensor3[1] = self.y + SENSOR_LENGTH * np.sin(angle_rad3)

# Classe pour gérer les boutons de l'interface
class Button:
    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = (200, 200, 200)
        self.font = pygame.font.Font(None, 32)
        
    def draw(self, screen):
        # Dessine le bouton et son texte
        pygame.draw.rect(screen, self.color, self.rect)
        text_surface = self.font.render(self.text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
        
    def is_clicked(self, pos):
        # Vérifie si le bouton est cliqué
        return self.rect.collidepoint(pos)

class Game:
    def __init__(self):
        # Initialisation de la fenêtre de jeu
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT + BUTTON_HEIGHT))
        pygame.display.set_caption("Self-Driving Car")
        self.clock = pygame.time.Clock()
        
        # Initialisation de la voiture et de l'environnement
        self.car = Car(WINDOW_WIDTH//2, WINDOW_HEIGHT//2)
        self.sand = torch.zeros((WINDOW_WIDTH, WINDOW_HEIGHT))  # Matrice du sable
        self.goal_x = 20  # Position X de l'objectif
        self.goal_y = WINDOW_HEIGHT - 20  # Position Y de l'objectif
        
        # Initialisation de l'IA et des paramètres d'apprentissage
        self.brain = Dqn(6, 3, 0.9)  # 6 entrées, 3 actions possibles
        self.action2rotation = [0, 20, -20]  # Rotations possibles
        self.last_reward = 0  # Dernière récompense
        self.scores = []  # Liste des scores
        self.last_distance = 0  # Dernière distance à l'objectif
        self.starting_time = default_timer()  # Temps de début
        
        # Variables pour le dessin
        self.drawing = False
        self.last_pos = None
        
        # Initialisation des boutons
        self.clear_button = Button(10, WINDOW_HEIGHT + 5, BUTTON_WIDTH, 30, "Clear")
        self.save_button = Button(120, WINDOW_HEIGHT + 5, BUTTON_WIDTH, 30, "Save")
        self.load_button = Button(230, WINDOW_HEIGHT + 5, BUTTON_WIDTH, 30, "Load")

    def handle_mouse_drawing(self, pos):
        # Gestion du dessin du sable avec la souris
        x, y = pos
        if y >= WINDOW_HEIGHT:  # Ne pas dessiner sur la zone des boutons
            return
            
        if self.last_pos is None:
            self.last_pos = pos
            return

        # Dessin de la ligne de sable
        pygame.draw.line(self.screen, (200, 180, 0), self.last_pos, pos, 10)
        
        # Mise à jour de la matrice de sable
        x1, y1 = self.last_pos
        x2, y2 = pos
        length = int(((x2-x1)**2 + (y2-y1)**2)**0.5)
        if length > 0:
            for i in range(length):
                t = i / length
                x = int(x1 * (1-t) + x2 * t)
                y = int(y1 * (1-t) + y2 * t)
                if 0 <= x < WINDOW_WIDTH and 0 <= y < WINDOW_HEIGHT:
                    self.sand[max(0, x-5):min(WINDOW_WIDTH, x+5),
                            max(0, y-5):min(WINDOW_HEIGHT, y+5)] = 1
        
        self.last_pos = pos

    def update(self):
        # Calcul du temps écoulé
        duration_time = default_timer() - self.starting_time
        
        # Calcul de l'orientation vers l'objectif
        xx = self.goal_x - self.car.x
        yy = self.goal_y - self.car.y
        
        # Calcul de l'orientation de la voiture
        velocity_angle = self.car.angle * np.pi / 180
        velocity_vector = np.array([np.cos(velocity_angle), np.sin(velocity_angle)])
        target_vector = np.array([xx, yy])
        target_norm = np.linalg.norm(target_vector)
        
        # Calcul de l'orientation relative à l'objectif
        if target_norm > 0:
            target_vector = target_vector / target_norm
            orientation = np.arccos(np.clip(np.dot(velocity_vector, target_vector), -1.0, 1.0)) / np.pi
        else:
            orientation = 0
        
        # Mise à jour des signaux des capteurs
        # Mise à jour des signaux des capteurs avec vérification des limites
        if int(self.car.sensor1[0]) >= 0 and int(self.car.sensor1[0]) < WINDOW_WIDTH and \
           int(self.car.sensor1[1]) >= 0 and int(self.car.sensor1[1]) < WINDOW_HEIGHT:
            self.car.signal1 = int(torch.sum(self.sand[
                max(0, int(self.car.sensor1[0])-10):min(WINDOW_WIDTH, int(self.car.sensor1[0])+10),
                max(0, int(self.car.sensor1[1])-10):min(WINDOW_HEIGHT, int(self.car.sensor1[1])+10)
            ])) / 400.  # Normalisation du signal entre 0 et 1
        else:
            self.car.signal1 = 1.  # Signal maximum si hors limites
            
        # Même chose pour le capteur 2
        if int(self.car.sensor2[0]) >= 0 and int(self.car.sensor2[0]) < WINDOW_WIDTH and \
           int(self.car.sensor2[1]) >= 0 and int(self.car.sensor2[1]) < WINDOW_HEIGHT:
            self.car.signal2 = int(torch.sum(self.sand[
                max(0, int(self.car.sensor2[0])-10):min(WINDOW_WIDTH, int(self.car.sensor2[0])+10),
                max(0, int(self.car.sensor2[1])-10):min(WINDOW_HEIGHT, int(self.car.sensor2[1])+10)
            ])) / 400.
        else:
            self.car.signal2 = 1.
            
        # Et pour le capteur 3
        if int(self.car.sensor3[0]) >= 0 and int(self.car.sensor3[0]) < WINDOW_WIDTH and \
           int(self.car.sensor3[1]) >= 0 and int(self.car.sensor3[1]) < WINDOW_HEIGHT:
            self.car.signal3 = int(torch.sum(self.sand[
                max(0, int(self.car.sensor3[0])-10):min(WINDOW_WIDTH, int(self.car.sensor3[0])+10),
                max(0, int(self.car.sensor3[1])-10):min(WINDOW_HEIGHT, int(self.car.sensor3[1])+10)
            ])) / 400.
        else:
            self.car.signal3 = 1.
        
        # Préparation des signaux pour l'IA
        last_signal = [
            self.car.signal1,    # Signal du capteur avant
            self.car.signal2,    # Signal du capteur droit
            self.car.signal3,    # Signal du capteur gauche
            orientation,         # Orientation vers l'objectif
            -orientation,        # Orientation inverse
            duration_time       # Temps écoulé
        ]
        
        # Mise à jour de l'IA et obtention de l'action
        action = self.brain.update(self.last_reward, last_signal)
        rotation = self.action2rotation[action]  # Conversion de l'action en rotation
        self.car.move(rotation)  # Application du mouvement
        
        # Calcul de la distance à l'objectif
        distance = np.sqrt((self.car.x - self.goal_x)**2 + (self.car.y - self.goal_y)**2)
        
        # Système de récompenses
        if int(self.car.x) < WINDOW_WIDTH and int(self.car.y) < WINDOW_HEIGHT and \
           self.sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = 1  # Ralentissement sur le sable
            self.last_reward = -1  # Punition pour être sur le sable
        else:
            self.car.velocity = 6  # Vitesse normale
            self.last_reward = -0.2  # Petite punition pour le temps
            if distance < self.last_distance:  # Récompense si on se rapproche de l'objectif
                self.last_reward = 0.1
                
        # Gestion des collisions avec les bords
        if self.car.x < 10:
            self.car.x = 10
            self.last_reward = -1
        if self.car.x > WINDOW_WIDTH - 10:
            self.car.x = WINDOW_WIDTH - 10
            self.last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            self.last_reward = -1
        if self.car.y > WINDOW_HEIGHT - 10:
            self.car.y = WINDOW_HEIGHT - 10
            self.last_reward = -1
            
        # Changement d'objectif si atteint
        if distance < 100:
            self.goal_x = WINDOW_WIDTH - self.goal_x
            self.goal_y = WINDOW_HEIGHT - self.goal_y
            self.starting_time = default_timer()
            
        # Pénalité de temps
        if duration_time > 10:
            self.last_reward = -0.2
            
        self.last_distance = distance
        self.scores.append(self.brain.score())
        
    def draw(self):
        # Nettoyage de l'écran
        self.screen.fill((255, 255, 255))
        
        # Dessin du sable
        sand_surface = pygame.surfarray.make_surface(self.sand.numpy())
        self.screen.blit(sand_surface, (0, 0))
        
        # Dessin de la voiture
        pygame.draw.circle(self.screen, (255, 0, 0), 
                         (int(self.car.x), int(self.car.y)), CAR_SIZE//2)
        
        # Dessin de l'indicateur de direction de la voiture
        direction_end = (
            int(self.car.x + CAR_SIZE * np.cos(self.car.angle * np.pi / 180)),
            int(self.car.y + CAR_SIZE * np.sin(self.car.angle * np.pi / 180))
        )
        pygame.draw.line(self.screen, (255, 0, 0),
                        (int(self.car.x), int(self.car.y)),
                        direction_end, 2)
        
        # Dessin des capteurs
        pygame.draw.line(self.screen, (0, 0, 255),
                        (int(self.car.x), int(self.car.y)),
                        (int(self.car.sensor1[0].item()), int(self.car.sensor1[1].item())))
        pygame.draw.line(self.screen, (0, 0, 255),
                        (int(self.car.x), int(self.car.y)),
                        (int(self.car.sensor2[0].item()), int(self.car.sensor2[1].item())))
        pygame.draw.line(self.screen, (0, 0, 255),
                        (int(self.car.x), int(self.car.y)),
                        (int(self.car.sensor3[0].item()), int(self.car.sensor3[1].item())))
        
        # Dessin de l'objectif
        pygame.draw.circle(self.screen, (0, 255, 0),
                         (int(self.goal_x), int(self.goal_y)), 10)
        
        # Dessin des boutons
        self.clear_button.draw(self.screen)
        self.save_button.draw(self.screen)
        self.load_button.draw(self.screen)
        
        # Mise à jour de l'affichage
        pygame.display.flip()
        
    def clear(self):
        # Effacement du sable
        self.sand = torch.zeros((WINDOW_WIDTH, WINDOW_HEIGHT))
        
    def save(self):
        # Sauvegarde du cerveau et affichage des scores
        self.brain.save()
        plt.plot(self.scores)
        plt.show()
        
    def load(self):
        # Chargement d'un cerveau précédemment sauvegardé
        self.brain.load()

def main():
    game = Game()
    running = True
    
    while running:
        # Boucle principale du jeu
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Gestion des clics de souris
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if game.clear_button.is_clicked(pos):
                    game.clear()
                elif game.save_button.is_clicked(pos):
                    game.save()
                elif game.load_button.is_clicked(pos):
                    game.load()
                else:
                    game.drawing = True
                    game.last_pos = pos
                    
            elif event.type == pygame.MOUSEBUTTONUP:
                game.drawing = False
                game.last_pos = None
                
            # Gestion du dessin du sable
            elif event.type == pygame.MOUSEMOTION and game.drawing:
                game.handle_mouse_drawing(pygame.mouse.get_pos())
                
        # Mise à jour et dessin du jeu
        game.update()
        game.draw()
        game.clock.tick(60)  # Limite à 60 FPS
        
    pygame.quit()

# Point d'entrée du programme
if __name__ == "__main__":
    main()