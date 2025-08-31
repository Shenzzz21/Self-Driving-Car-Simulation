"""
Agent module - Self-driving car with reinforcement learning
Implements Sense-Think-Act-Reflect paradigm with A* path following
"""

import pygame
import random
import math
from collections import deque
from pathfinding import AStar


class SelfDrivingAgent:
    def __init__(self, environment):
        self.env = environment
        self.position = list(self.env.start_pos)
        self.heading = 0  # 0=right, 1=down, 2=left, 3=up

        # Q-Learning parameters
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.3  # Exploration rate

        # Actions: 0=forward, 1=turn_right, 2=turn_left, 3=backward
        self.actions = [0, 1, 2, 3]
        self.action_names = ["Forward", "Turn Right", "Turn Left", "Backward"]

        # Direction vectors: right, down, left, up
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        # Path planning
        self.pathfinder = AStar(environment)
        self.planned_path = deque()

        # State and rewards
        self.total_reward = 0
        self.steps = 0
        self.max_steps = 500
        self.prev_dest_dist = None

        # Load car image
        self._load_car_image()

        # Parameters for replanning
        self.replan_on_deviation_threshold = 2.0
        self.replan_every_n_steps = 30
        self._last_replan_step = 0

    def update_environment_ref(self, environment):
        """Call this when the environment changes (new obstacles/destination)."""
        self.env = environment
        self.pathfinder = AStar(environment)
        self.planned_path.clear()
        # reset prev distance so reward shaping recalculates properly
        self.prev_dest_dist = None

    def _load_car_image(self):
        """Load and prepare car image"""
        try:
            self.car_img = pygame.image.load("assets/car.png")
            self.car_img = pygame.transform.scale(self.car_img, (self.env.grid_size-2, self.env.grid_size-2))
        except:
            self.car_img = pygame.Surface((self.env.grid_size-2, self.env.grid_size-2), pygame.SRCALPHA)
            pygame.draw.rect(self.car_img, (0, 100, 255), (0, 0, self.env.grid_size-2, self.env.grid_size-2))

        # Create rotated versions for different headings
        self.car_images = []
        for angle in [0, 90, 180, 270]:  # right, down, left, up
            rotated = pygame.transform.rotate(self.car_img, -angle)
            self.car_images.append(rotated.convert_alpha())

    def _maybe_replan(self):
        """Replan path if deviation is large or periodically to react to obstacles."""
        if (self.steps - self._last_replan_step) >= self.replan_every_n_steps:
            self._last_replan_step = self.steps
            path = self.pathfinder.find_path(tuple(self.position), self.env.destination)
            if path:
                self.planned_path = deque(path[1:])  # skip current position
            return

        if self.planned_path:
            x, y = self.position
            # safety: if planned path includes invalid points (obstacle landed on it), force replan
            for px, py in list(self.planned_path):
                if (px, py) in self.env.obstacles:
                    path = self.pathfinder.find_path(tuple(self.position), self.env.destination)
                    if path:
                        self.planned_path = deque(path[1:])
                    self._last_replan_step = self.steps
                    return

            min_dist = min(math.sqrt((x - px)**2 + (y - py)**2) for px, py in self.planned_path)
            if min_dist > self.replan_on_deviation_threshold:
                path = self.pathfinder.find_path(tuple(self.position), self.env.destination)
                if path:
                    self.planned_path = deque(path[1:])
                self._last_replan_step = self.steps

    def get_state(self):
        """SENSE: Get current state representation"""
        x, y = self.position

        dest_dist = math.sqrt((x - self.env.destination[0])**2 + (y - self.env.destination[1])**2)

        obstacles = self.env.get_obstacles_in_radius(tuple(self.position), 5)
        min_obstacle_dist = 5 if not obstacles else min(obs[2] for obs in obstacles)

        # Path deviation
        if not self.planned_path:
            path_dev = 0
        else:
            min_path_dist = min(math.sqrt((x - px)**2 + (y - py)**2) for px, py in self.planned_path)
            path_dev = min(min_path_dist, 3)

        state = (
            min(int(dest_dist), 20),
            min(int(min_obstacle_dist), 5),
            self.heading,
            int(path_dev)
        )
        return state

    def choose_action(self, state):
        """THINK: Follow A* path if available, otherwise use epsilon-greedy Q-learning"""
        # If path exists, move to next step
        if self.planned_path:
            next_pos = self.planned_path[0]
            dx = next_pos[0] - self.position[0]
            dy = next_pos[1] - self.position[1]
            if dx == 1 and dy == 0:
                return 0 if self.heading == 0 else 1  # forward or turn
            elif dx == -1 and dy == 0:
                return 0 if self.heading == 2 else 1
            elif dx == 0 and dy == 1:
                return 0 if self.heading == 1 else 1
            elif dx == 0 and dy == -1:
                return 0 if self.heading == 3 else 1

        # Otherwise, Q-learning fallback
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(self.actions)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return self.actions[self.q_table[state].index(max(self.q_table[state]))]

    def execute_action(self, action):
        """ACT: Execute chosen action"""
        old_pos = tuple(self.position)

        if action == 0:  # forward
            dx, dy = self.directions[self.heading]
            new_pos = (self.position[0]+dx, self.position[1]+dy)
            if self.env.is_valid_position(new_pos):
                self.position = list(new_pos)
                if self.planned_path and self.planned_path[0] == new_pos:
                    self.planned_path.popleft()
        elif action == 1:  # turn right
            self.heading = (self.heading + 1) % 4
        elif action == 2:  # turn left
            self.heading = (self.heading - 1) % 4
        elif action == 3:  # backward
            dx, dy = self.directions[self.heading]
            new_pos = (self.position[0]-dx, self.position[1]-dy)
            if self.env.is_valid_position(new_pos):
                self.position = list(new_pos)

        return tuple(self.position) != old_pos

    def calculate_reward(self, old_state, action, moved):
        """Calculate reward"""
        reward = -1  # time penalty

        if tuple(self.position) == self.env.destination:
            reward += 100
            return reward, True

        if tuple(self.position) in self.env.obstacles:
            reward -= 100
            return reward, True

        if not self.env.is_road(tuple(self.position)):
            reward -= 10

        # reward for progress
        dist = math.sqrt((self.position[0]-self.env.destination[0])**2 + (self.position[1]-self.env.destination[1])**2)
        if self.prev_dest_dist is not None and dist < self.prev_dest_dist:
            reward += 2
        self.prev_dest_dist = dist

        return reward, False

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0.0]*len(self.actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0]*len(self.actions)
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state])
        self.q_table[state][action] = current_q + self.learning_rate * (reward + self.discount_factor*max_next_q - current_q)

    def step(self):
        """Main step function"""
        if self.steps == 0:
            path = self.pathfinder.find_path(tuple(self.position), self.env.destination)
            if path:
                self.planned_path = deque(path[1:])
            self.prev_dest_dist = math.sqrt((self.position[0]-self.env.destination[0])**2 +
                                            (self.position[1]-self.env.destination[1])**2)

        self._maybe_replan()

        state = self.get_state()
        action = self.choose_action(state)
        moved = self.execute_action(action)
        reward, done = self.calculate_reward(state, action, moved)
        self.total_reward += reward
        self.steps += 1
        next_state = self.get_state()
        self.update_q_table(state, action, reward, next_state)

        if self.steps >= self.max_steps:
            done = True
        return done

    def reset(self):
        self.position = list(self.env.start_pos)
        self.heading = 0
        self.total_reward = 0
        self.steps = 0
        self.planned_path.clear()
        self.prev_dest_dist = None
        # slightly reduce exploration (annealing)
        self.epsilon = max(0.05, self.epsilon*0.995)

    def render(self, screen):
        """Render agent and path"""
        if len(self.planned_path) > 1:
            path_points = [self.env.grid_to_pixel(pos) for pos in self.planned_path]
            pygame.draw.lines(screen, (255, 255, 0), False, path_points, 2)

        car_pixel = self.env.grid_to_pixel(tuple(self.position))
        car_rect = self.car_images[self.heading].get_rect()
        car_rect.center = car_pixel
        screen.blit(self.car_images[self.heading], car_rect)
