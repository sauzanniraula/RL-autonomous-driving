import os
import cv2
import pygame
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process_img(image, dim_x=128, dim_y=128):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    dim = (dim_x, dim_y)
    resized_img = cv2.resize(array, dim, interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    # Returns float 0.0 to 1.0
    return img_gray / 255.0

def get_speed(vehicle):
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

def correct_yaw(x):
    return(((x%360) + 360) % 360)

def create_folders(folder_names):
    for directory in folder_names:
        if not os.path.exists(directory):
            os.makedirs(directory)

class MetricsLogger:
    def __init__(self, save_dir='plots'):
        self.save_dir = save_dir
        create_folders([save_dir, os.path.join(save_dir, 'episodes')])
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        self.epsilons = []
        self.collision_history = []
        self.episode_avg_speeds = []
        self.episode_distances = []
        
        self.reset_episode()

    def reset_episode(self):
        self.current_ep_trajectory = []
        self.current_ep_controls = []
        self.current_ep_speeds = []
        self.current_ep_limits = []

    def log_step(self, location, control, speed, speed_limit):
        self.current_ep_trajectory.append((location.x, location.y))
        self.current_ep_controls.append((control.steer, control.throttle, control.brake))
        self.current_ep_speeds.append(speed)
        self.current_ep_limits.append(speed_limit)

    # --- FIX IS HERE: Added 'length' parameter back ---
    def log_episode(self, reward, length, avg_speed, epsilon, distance, collision):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length) # Store the steps (duration)
        self.episode_avg_speeds.append(avg_speed)
        self.epsilons.append(epsilon)
        self.episode_distances.append(distance)
        self.collision_history.append(collision)
        
    def log_loss(self, loss):
        self.training_losses.append(loss)

    def save_training_plots(self):
        try:
            plt.close('all') # FORCE CLEANUP to stop memory leaks
            fig, axs = plt.subplots(3, 2, figsize=(15, 15))
            
            # Reward
            axs[0, 0].plot(self.episode_rewards, color='blue')
            axs[0, 0].set_title('Reward')
            
            # Loss
            axs[0, 1].plot(self.training_losses, color='orange')
            axs[0, 1].set_yscale('log')
            axs[0, 1].set_title('Loss')

            # Epsilon
            axs[1, 0].plot(self.epsilons, color='purple')
            axs[1, 0].set_title('Epsilon')

            # Crash Rate
            collisions = np.array(self.collision_history)
            if len(collisions) > 0:
                window = 50
                rolling = [np.mean(collisions[max(0, i-window):i+1]) for i in range(len(collisions))]
                axs[1, 1].plot(rolling, color='red')
                axs[1, 1].set_title('Crash Rate')

            # Duration (Survival Time)
            axs[2, 0].plot(self.episode_lengths, color='green')
            axs[2, 0].set_title('Episode Duration (Steps)')

            # Distance
            axs[2, 1].plot(self.episode_distances, label='Distance', color='cyan')
            axs[2, 1].set_title('Distance Traveled')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'training_metrics.png'))
            plt.close(fig)
        except Exception as e:
            print(f"Plotting error: {e}")

    def save_episode_plots(self, ep):
        try:
            plt.close('all')
            fig, axs = plt.subplots(2, 2, figsize=(16, 10))
            
            traj = np.array(self.current_ep_trajectory)
            if len(traj) > 0:
                axs[0, 0].plot(traj[:, 0], traj[:, 1])
                axs[0, 0].set_title('Trajectory')
                axs[0, 0].axis('equal')

            axs[0, 1].plot(self.current_ep_speeds, color='purple')
            axs[0, 1].set_title('Speed')

            controls = np.array(self.current_ep_controls)
            if len(controls) > 0:
                axs[1, 0].plot(controls[:, 0], label='Steer')
                axs[1, 0].legend()
                axs[1, 0].set_title('Steering')

            axs[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'episodes', f'episode_{ep}_analysis.png'))
            plt.close(fig)
            self.reset_episode()
        except Exception as e:
            print(f"Ep plot error: {e}")