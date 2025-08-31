"""
Pathfinding module - A* algorithm implementation
"""

import heapq
import math

class AStar:
    def __init__(self, environment):
        self.env = environment
    
    def heuristic(self, pos1, pos2):
        """Calculate Manhattan distance heuristic"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_neighbors(self, pos):
        """Get valid neighboring positions"""
        x, y = pos
        neighbors = []
        
        # 4-directional movement (up, down, left, right)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_pos = (x + dx, y + dy)
            if self.env.is_valid_position(new_pos):
                neighbors.append(new_pos)
        
        return neighbors
    
    def reconstruct_path(self, came_from, current):
        """Reconstruct path from goal to start"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # Reverse to get start-to-goal path
    
    def find_path(self, start, goal):
        """
        Find shortest path using A* algorithm
        Following the pseudocode from the proposal
        """
        # INITIALIZE open_set with the starting node
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        # INITIALIZE cost and heuristic estimates for nodes
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        # WHILE open_set is not empty DO
        while open_set:
            # SELECT node with the lowest estimated total cost
            current_f, current = heapq.heappop(open_set)
            
            # IF node is the goal THEN RETURN the reconstructed path
            if current == goal:
                return self.reconstruct_path(came_from, current)
            
            # FOR each neighbor of the current node DO
            for neighbor in self.get_neighbors(current):
                # CALCULATE the cost to reach the neighbor
                tentative_g_score = g_score[current] + 1
                
                # IF this path to neighbor is better than any previous path THEN
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # RECORD the new best path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    
                    # IF neighbor is not in open_set THEN ADD neighbor to open_set
                    if not any(neighbor == item[1] for item in open_set):
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found
        return []