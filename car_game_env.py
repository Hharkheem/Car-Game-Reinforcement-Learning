import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random

class CarGameEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None):
        super(CarGameEnv, self).__init__()

        self.render_mode = render_mode
        self.clock = pygame.time.Clock()

        # Pygame setup
        pygame.init()
        self.width = 500
        self.height = 500
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Car RL Environment')

        # Game state
        self.lanes = [150, 250, 350]
        self.player_lane = 1
        self.player_y = 400
        self.speed = 1
        self.score = 0
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # left, stay, right
        self.observation_space = spaces.Box(low=0, high=500, shape=(4,), dtype=np.float32)
        self.vehicles = []


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.player_lane = 1
        self.player_y = 400
        self.speed = 1
        self.score = 0
        self.vehicles = []
        return self._get_obs(), {}

    def step(self, action):
        # Apply action
        if action == 0 and self.player_lane > 0:
            self.player_lane -= 1
        elif action == 2 and self.player_lane < 2:
            self.player_lane += 1

        reward = 0.1  # base reward for surviving
        done = False

        # Move vehicles and check passed
        new_vehicles = []
        for lane, y in self.vehicles:
            y += self.speed
            if y < self.height:
                new_vehicles.append((lane, y))
            else:
                reward += 2  # reward for passing
                self.score += 1
                if self.score % 5 == 0:
                    self.speed += 1
        self.vehicles = new_vehicles

        # Add new vehicle occasionally
        if len(self.vehicles) < 2 and random.random() < 0.1:
            new_lane = random.randint(0, 2)
            self.vehicles.append((new_lane, -50))

        # Penalize proximity
        for lane, y in self.vehicles:
            if lane == self.player_lane:
                distance = abs(self.player_y - y)
                if 0 < distance < 100:
                    reward -= 1.0 / distance  # closer = higher penalty

        # Check crash
        for lane, y in self.vehicles:
            if lane == self.player_lane and abs(self.player_y - y) < 40:
                reward -= 20
                done = True
                break

        return self._get_obs(), reward, done, False, {}

    def render(self):
        self.screen.fill((76, 208, 56))
        pygame.draw.rect(self.screen, (100, 100, 100), (100, 0, 300, self.height))

        # Draw player car
        pygame.draw.rect(self.screen, (0, 0, 255), (self.lanes[self.player_lane] - 20, self.player_y, 40, 60))

        # Draw other vehicles
        for lane, y in self.vehicles:
            pygame.draw.rect(self.screen, (255, 0, 0), (self.lanes[lane] - 20, y, 40, 60))

        if self.render_mode == 'human':
            pygame.display.flip()
            self.clock.tick(60)
        elif self.render_mode == 'rgb_array':
            return np.transpose(
                pygame.surfarray.array3d(pygame.display.get_surface()), axes=(1, 0, 2)
            )
    
    # I am trying to render the game in pygame format so it shows the cars in the picture form
    # def render(self):
    #     if self.render_mode == "human":
    #         if self.screen is None:
    #             pygame.init()
    #             self.screen = pygame.display.set_mode((self.width, self.height))
    #             self.clock = pygame.time.Clock()
    #         self._draw(self.screen)
    #         pygame.display.flip()
    #         # self.clock.tick(FPS)

    #     elif self.render_mode == "rgb_array":
    #         if self.surface is None:
    #             pygame.init()
    #             self.surface = pygame.Surface((self.width, self.height))
    #         self._draw(self.surface)
    #         return np.transpose(
    #             pygame.surfarray.array3d(self.surface), axes=(1, 0, 2)
    #         )

    def close(self):
        pygame.quit()

    def _get_obs(self):
        obs = [self.player_lane]
        dists = [500, 500, 500]
        for lane in range(3):
            for v_lane, y in self.vehicles:
                if v_lane == lane and y >= 0:
                    dists[lane] = min(dists[lane], abs(self.player_y - y))
        obs.extend(dists)
        return np.array(obs, dtype=np.float32)


