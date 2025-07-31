import pygame
from pygame.locals import *
import random
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

pygame.init()
plt.ion()  # Enable interactive plotting

# Load trained model
model = DQN.load("best_model/best_model")

# Window setup
width = 500
height = 500
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Car Game - AI Controlled')

# Colors
gray = (100, 100, 100)
green = (76, 208, 56)
red = (200, 0, 0)
white = (255, 255, 255)
yellow = (255, 232, 0)

# Road settings
road_width = 300
marker_width = 10
marker_height = 50

left_lane = 150
center_lane = 250
right_lane = 350
lanes = [left_lane, center_lane, right_lane]

road = (100, 0, road_width, height)
left_edge_marker = (95, 0, marker_width, height)
right_edge_marker = (395, 0, marker_width, height)

# Game state
lane_marker_move_y = 0
player_x = center_lane
player_y = 400
clock = pygame.time.Clock()
fps = 60
gameover = False
speed = 1
score = 0

# Logging setup
log_file = 'scores.csv'
if not os.path.exists(log_file):
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Score'])

episode_count = 0
scores = []

# Vehicle class
class Vehicle(pygame.sprite.Sprite):
    def __init__(self, image, x, y):
        pygame.sprite.Sprite.__init__(self)
        image_scale = 45 / image.get_rect().width
        new_width = int(image.get_rect().width * image_scale)
        new_height = int(image.get_rect().height * image_scale)
        self.image = pygame.transform.scale(image, (new_width, new_height))
        self.rect = self.image.get_rect()
        self.rect.center = [x, y]

class PlayerVehicle(Vehicle):
    def __init__(self, x, y):
        image = pygame.image.load('images/car.png')
        super().__init__(image, x, y)

# Sprites
player_group = pygame.sprite.Group()
vehicle_group = pygame.sprite.Group()
player = PlayerVehicle(player_x, player_y)
player_group.add(player)

# Vehicle images
image_filenames = ['pickup_truck.png', 'semi_trailer.png', 'taxi.png', 'van.png']
vehicle_images = [pygame.image.load('images/' + fname) for fname in image_filenames]

crash = pygame.image.load('images/crash.png')
crash_rect = crash.get_rect()

# Main game loop
running = True
while running:
    clock.tick(fps)
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    # === RL Agent Logic ===
    player_lane = lanes.index(player.rect.center[0])
    dists = [500, 500, 500]
    for vehicle in vehicle_group:
        v_lane = lanes.index(vehicle.rect.center[0])
        dy = abs(player.rect.y - vehicle.rect.y)
        if vehicle.rect.y > 0 and dy < dists[v_lane]:
            dists[v_lane] = dy
    obs = np.array([player_lane] + dists, dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)

    # Apply action
    if action == 0 and player.rect.center[0] > left_lane:
        player.rect.x -= 100
    elif action == 2 and player.rect.center[0] < right_lane:
        player.rect.x += 100

    # Check for side collision
    for vehicle in vehicle_group:
        if pygame.sprite.collide_rect(player, vehicle):
            gameover = True
            crash_rect.center = [(player.rect.center[0] + vehicle.rect.center[0]) / 2,
                                 (player.rect.center[1] + vehicle.rect.center[1]) / 2]

    # === Drawing ===
    screen.fill(green)
    pygame.draw.rect(screen, gray, road)
    pygame.draw.rect(screen, yellow, left_edge_marker)
    pygame.draw.rect(screen, yellow, right_edge_marker)

    lane_marker_move_y += speed * 2
    if lane_marker_move_y >= marker_height * 2:
        lane_marker_move_y = 0
    for y in range(marker_height * -2, height, marker_height * 2):
        pygame.draw.rect(screen, white, (left_lane + 45, y + lane_marker_move_y, marker_width, marker_height))
        pygame.draw.rect(screen, white, (center_lane + 45, y + lane_marker_move_y, marker_width, marker_height))

    player_group.draw(screen)

    if len(vehicle_group) < 2:
        add_vehicle = True
        for vehicle in vehicle_group:
            if vehicle.rect.top < vehicle.rect.height * 1.5:
                add_vehicle = False
        if add_vehicle:
            lane = random.choice(lanes)
            image = random.choice(vehicle_images)
            vehicle = Vehicle(image, lane, height / -2)
            vehicle_group.add(vehicle)

    for vehicle in vehicle_group:
        vehicle.rect.y += speed
        if vehicle.rect.top >= height:
            vehicle.kill()
            score += 1
            if score % 5 == 0:
                speed += 1

    vehicle_group.draw(screen)

    font = pygame.font.Font(pygame.font.get_default_font(), 16)
    text = font.render('Score: ' + str(score), True, white)
    screen.blit(text, (10, 450))

    if pygame.sprite.spritecollide(player, vehicle_group, True):
        gameover = True
        crash_rect.center = [player.rect.center[0], player.rect.top]

    if gameover:
        def restart_game():
            global score, speed, gameover, vehicle_group, player
            score = 0
            speed = 1
            gameover = False
            vehicle_group.empty()
            player.rect.center = [player_x, player_y]
            
        screen.blit(crash, crash_rect)
        pygame.draw.rect(screen, red, (0, 50, width, 100))
        text = font.render('Game over. Press R to restart or ESC to quit.', True, white)
        text_rect = text.get_rect(center=(width // 2, 100))
        screen.blit(text, text_rect)
        pygame.display.update()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                    waiting = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                        waiting = False
                    elif event.key == K_r:
                        restart_game()
                        waiting = False
                    

        # === Score logging ===
        episode_count += 1
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode_count, score])
        scores.append(score)

        # # Plot live performance
        # plt.clf()
        # plt.title("Agent Score Per Episode")
        # plt.xlabel("Episode")
        # plt.ylabel("Score")
        # plt.plot(scores, label="Score")
        # if len(scores) >= 5:
        #     avg = np.convolve(scores, np.ones(5)/5, mode='valid')
        #     plt.plot(range(4, len(scores)), avg, label="5-ep Moving Avg")
        # plt.legend()
        # plt.pause(0.001)

    pygame.display.update()
    

    while gameover:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                gameover = False
                running = False
                break

        if not running:
            break

pygame.quit()
