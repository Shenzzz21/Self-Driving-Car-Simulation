"""
Environment module - 2D road layout with lanes and obstacles
Includes A* pathfinding support.
"""

import pygame
import random
import math

class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid_size = 20
        self.grid_width = width // self.grid_size
        self.grid_height = height // self.grid_size

        # Road layout
        self.road_map = self._create_road_layout()

        # Start and destination
        self.start_pos = (2, self.grid_height // 2)
        self.destination = (self.grid_width - 3, self.grid_height // 2)

        # Obstacles
        self.obstacles = self._place_obstacles(15)

        # Load images
        self._load_images()

        # Car current position
        self.car_position = self.start_pos

    def _load_images(self):
        """Load cone image"""
        try:
            self.cone_img = pygame.image.load("assets/cone.png")
            self.cone_img = pygame.transform.scale(self.cone_img, (self.grid_size-2, self.grid_size-2))
        except:
            self.cone_img = pygame.Surface((self.grid_size-2, self.grid_size-2), pygame.SRCALPHA)
            # draw a simple cone
            points = [(self.grid_size//2-1, 2), (2, self.grid_size-4), (self.grid_size-4, self.grid_size-4)]
            pygame.draw.polygon(self.cone_img, (255, 69, 0), points)

    def _create_road_layout(self):
        road_map = [[0 for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        road_start = self.grid_height // 2 - 2
        road_end = self.grid_height // 2 + 3
        for y in range(road_start, road_end):
            for x in range(self.grid_width):
                road_map[y][x] = 1
        # Vertical roads
        for road_x in [self.grid_width // 3, 2 * self.grid_width // 3]:
            for y in range(self.grid_height):
                if y < road_start or y >= road_end:
                    for x in range(road_x-1, road_x+2):
                        if 0 <= x < self.grid_width:
                            road_map[y][x] = 1
        return road_map

    def _place_obstacles(self, num_obstacles=15):
        obstacles = set()
        attempts = 0
        max_attempts = num_obstacles * 50
        while len(obstacles) < num_obstacles and attempts < max_attempts:
            attempts += 1
            x = random.randint(0, self.grid_width - 1)
            y = random.randint(0, self.grid_height - 1)
            # place obstacles only on road cells, avoid start and destination
            if (self.road_map[y][x] == 1 and
                (x, y) not in obstacles and
                (x, y) != self.start_pos and
                (x, y) != self.destination):
                obstacles.add((x, y))
        return obstacles

    def regenerate_obstacles(self, num_obstacles=25):
        """Public API to regenerate obstacles with desired density.
        Returns the new obstacle set for convenience.
        """
        self.obstacles = self._place_obstacles(num_obstacles)
        return self.obstacles

    def is_valid_position(self, pos):
        x, y = pos
        return (0 <= x < self.grid_width and 0 <= y < self.grid_height and
                self.road_map[y][x] == 1 and pos not in self.obstacles)

    def is_road(self, pos):
        x, y = pos
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height and self.road_map[y][x] == 1

    def get_obstacles_in_radius(self, pos, radius):
        nearby = []
        x, y = pos
        for ox, oy in self.obstacles:
            dist = math.sqrt((x-ox)**2 + (y-oy)**2)
            if dist <= radius:
                nearby.append((ox, oy, dist))
        return nearby

    def pixel_to_grid(self, pixel_pos):
        x, y = pixel_pos
        return (x // self.grid_size, y // self.grid_size)

    def grid_to_pixel(self, grid_pos):
        x, y = grid_pos
        return (x*self.grid_size + self.grid_size//2, y*self.grid_size + self.grid_size//2)

    def set_destination(self, pos):
        if self.is_valid_position(pos):
            self.destination = pos
            return True
        return False

    # A* pathfinding (kept for backward compatibility)
    def a_star_search(self, start, goal):
        import heapq
        def heuristic(a, b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start:0}
        f_score = {start:heuristic(start, goal)}
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                # reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            x, y = current
            for dx, dy in [(1,0),(0,1),(-1,0),(0,-1)]:
                neighbor = (x+dx, y+dy)
                if not self.is_valid_position(neighbor):
                    continue
                tentative_g = g_score[current]+1
                if neighbor not in g_score or tentative_g<g_score[neighbor]:
                    came_from[neighbor]=current
                    g_score[neighbor]=tentative_g
                    f_score[neighbor]=tentative_g+heuristic(neighbor, goal)
                    heapq.heappush(open_set,(f_score[neighbor], neighbor))
        return []

    def render(self, screen):
        # Draw road/off-road
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                rect = (x*self.grid_size, y*self.grid_size, self.grid_size, self.grid_size)
                if self.road_map[y][x]==1:
                    pygame.draw.rect(screen, (80,80,80), rect)
                    if x%3==0:
                        pygame.draw.line(screen,(255,255,255),
                                         (x*self.grid_size+self.grid_size//2, y*self.grid_size),
                                         (x*self.grid_size+self.grid_size//2, y*self.grid_size+self.grid_size))
                else:
                    pygame.draw.rect(screen, (34,139,34), rect)
        # Draw obstacles
        for ox, oy in self.obstacles:
            screen.blit(self.cone_img, (ox*self.grid_size+1, oy*self.grid_size+1))
        # Draw destination
        dest_pixel = self.grid_to_pixel(self.destination)
        pygame.draw.circle(screen, (0,255,0), dest_pixel, self.grid_size//2)
