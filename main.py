"""
Self-Driving Car Simulation - Main Demonstration File
"""

import pygame
import sys
import random
from environment import Environment
from agent import SelfDrivingAgent

def main():
    # Initialize Pygame
    pygame.init()
    
    # Constants
    WINDOW_WIDTH = 1000
    WINDOW_HEIGHT = 700
    FPS = 30
    
    # Create display
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Self-Driving Car - AI Demonstration")
    clock = pygame.time.Clock()
    
    # Create environment and agent
    env = Environment(WINDOW_WIDTH, WINDOW_HEIGHT)
    agent = SelfDrivingAgent(env)
    
    episode = 1

    # Function to generate a new random destination
    def random_destination(prev_dest=None):
        attempts = 0
        while True:
            attempts += 1
            x = random.randint(0, env.grid_width-1)
            y = random.randint(0, env.grid_height-1)
            if env.is_valid_position((x, y)) and (x, y) != env.start_pos and (x, y) != prev_dest:
                return (x, y)
            if attempts > 1000:
                # fallback to center-ish cell
                return (env.grid_width//2, env.grid_height//2)
    
    # Initialize first destination and obstacles
    prev_dest = None
    new_dest = random_destination(prev_dest)
    env.set_destination(new_dest)
    env.regenerate_obstacles(num_obstacles=25)
    agent.update_environment_ref(env)
    print(f"Starting Episode {episode}, Destination: {env.destination}, Obstacles: {len(env.obstacles)}")

    running = True
    while running:
        step_reward = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # User can manually set destination
                mouse_pos = pygame.mouse.get_pos()
                grid_pos = env.pixel_to_grid(mouse_pos)
                if env.is_valid_position(grid_pos):
                    env.set_destination(grid_pos)
                    env.regenerate_obstacles(num_obstacles=25)
                    agent.update_environment_ref(env)
                    agent.reset()
                    episode += 1
                    prev_dest = grid_pos
                    print(f"\nNew destination set manually! Episode {episode}, Destination: {grid_pos}, Obstacles: {len(env.obstacles)}")

        # Agent step
        done = agent.step()

        # calculate immediate step reward (difference in total_reward since last step)
        # store previous total to compute step reward
        if 'last_total' not in locals():
            last_total = agent.total_reward
        step_reward = agent.total_reward - last_total
        last_total = agent.total_reward

        # Print step reward to terminal (lightweight)
        print(f"Episode {episode} | Step {agent.steps} | Step reward: {step_reward:.1f} | Total reward: {agent.total_reward:.1f}", end='\r')

        if done:
            # Episode finished: print summary, then make a new destination and regenerate obstacles
            print()  # newline to clear the carriage return line
            print(f"Episode {episode} completed in {agent.steps} steps. Total reward: {agent.total_reward:.1f}")
            prev_dest = env.destination
            new_dest = random_destination(prev_dest)
            env.set_destination(new_dest)
            env.regenerate_obstacles(num_obstacles=25 + min(30, episode))  # increase density with episodes
            agent.update_environment_ref(env)
            agent.reset()
            episode += 1
            print(f"New Destination: {new_dest}. Obstacles regenerated: {len(env.obstacles)}. Starting Episode {episode}")

        # Render everything
        screen.fill((50, 50, 50))  # Dark background
        env.render(screen)
        agent.render(screen)

        # Display episode info
        font = pygame.font.Font(None, 36)
        text = font.render(f"Episode: {episode} | Reward: {agent.total_reward:.1f} | Destination: {env.destination}", True, (255, 255, 255))
        screen.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
