import glob
import os
import sys
import numpy as np
from collections import deque

# --- PATH SETUP ---
# Ensure this matches your installation
carla_root = 'C:/Carla/CARLA_0.9.16/PythonAPI/carla/dist/'
try:
    egg_path = glob.glob(carla_root + 'carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0]
    sys.path.append(egg_path)
except IndexError:
    pass
# ------------------

import carla
import random
import time

from synch_mode import CarlaSyncMode
from controllers import PIDLongitudinalController
from utils import *

random.seed(78)

class SimEnv(object):
    def __init__(self, 
        visuals=False, # We don't use a separate window anymore
        target_speed = 30,
        max_iter = 4000,
        start_buffer = 10,
        train_freq = 1,
        save_freq = 200,
        start_ep = 0,
        max_dist_from_waypoint = 20,
        lambda_collision = 50
    ) -> None:
        self.visuals = visuals 
        
        print("Connecting to CARLA Server...")
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)

        print("Loading World Town02_Opt...")
        self.world = self.client.load_world('Town02_Opt')
        self.world.unload_map_layer(carla.MapLayer.Decals)
        self.world.unload_map_layer(carla.MapLayer.Foliage)
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        self.world.unload_map_layer(carla.MapLayer.Particles)
        self.world.unload_map_layer(carla.MapLayer.Props)
        self.world.unload_map_layer(carla.MapLayer.StreetLights)
        
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.blueprint_library = self.world.get_blueprint_library()
        
        # --- REVERTED TO ORIGINAL VEHICLE (Slower/Stable) ---
        self.vehicle_blueprint = self.blueprint_library.find('vehicle.tesla.model3')

        self.global_t = 0
        self.target_speed = target_speed 
        self.max_iter = max_iter
        self.start_buffer = start_buffer
        self.train_freq = train_freq
        self.save_freq = save_freq
        self.start_ep = start_ep
        self.lambda_collision = lambda_collision
        self.max_dist_from_waypoint = max_dist_from_waypoint
        self.start_train = self.start_ep + self.start_buffer
        
        self.total_rewards = 0
        self.average_rewards_list = []
        self.frame_stack = deque(maxlen=4)
        
        self.logger = MetricsLogger(save_dir='plots')
    
    def create_actors(self):
        self.actor_list = []
        self.vehicle = self.world.spawn_actor(self.vehicle_blueprint, random.choice(self.spawn_points))
        self.actor_list.append(self.vehicle)

        # 1. The "Brain" Camera (What the AI sees)
        self.camera_rgb = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_rgb)

        # 2. The "Visual" Camera (For the Spectator View)
        # Positioned BEHIND and ABOVE (Third Person)
        self.camera_rgb_vis = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_rgb_vis)

        self.collision_sensor = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.collision'),
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.actor_list.append(self.collision_sensor)
        self.speed_controller = PIDLongitudinalController(self.vehicle)
    
    def reset(self):
        for actor in self.actor_list:
            if actor.is_alive:
                actor.destroy()
        self.frame_stack.clear()
    
    def _get_stacked_state(self):
        return np.array(self.frame_stack)

    def generate_episode(self, model, replay_buffer, ep, action_map=None, eval=True):
        with CarlaSyncMode(self.world, self.camera_rgb, self.camera_rgb_vis, self.collision_sensor, fps=30) as sync_mode:
            counter = 0
            snapshot, image_rgb, image_rgb_vis, collision = sync_mode.tick(timeout=2.0)

            if snapshot is None or image_rgb is None:
                self.reset()
                return None

            image = process_img(image_rgb)
            for _ in range(4):
                self.frame_stack.append(image)
            next_state = self._get_stacked_state()

            episode_reward = 0
            episode_speed_sum = 0
            start_location = self.vehicle.get_location()
            
            # --- SPECTATOR SETUP ---
            spectator = self.world.get_spectator()
            
            print(f"\n--- Starting Episode {ep} (Epsilon: {model.current_eps:.4f}) ---")

            while True:
                # --- LOCK CAMERA TO CAR (SPECTATOR MODE) ---
                spectator.set_transform(self.camera_rgb_vis.get_transform())

                vehicle_location = self.vehicle.get_location()
                waypoint = self.world.get_map().get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
                
                speed = get_speed(self.vehicle)
                # Safety check: if get_speed_limit fails, default to 30
                try:
                    speed_limit = self.vehicle.get_speed_limit()
                except:
                    speed_limit = 30.0

                state = next_state
                counter += 1
                self.global_t += 1
                episode_speed_sum += speed

                action = model.select_action(state, eval=eval)
                steer = action
                if action_map is not None:
                    steer = action_map[action]

                control = self.speed_controller.run_step(self.target_speed)
                control.steer = steer
                self.vehicle.apply_control(control)
                
                self.logger.log_step(vehicle_location, control, speed, speed_limit)

                snapshot, image_rgb, image_rgb_vis, collision = sync_mode.tick(timeout=2.0)
                cos_yaw_diff, dist, collision = get_reward_comp(self.vehicle, waypoint, collision)
                reward = reward_value(cos_yaw_diff, dist, collision, lambda_3=self.lambda_collision)

                if snapshot is None or image_rgb is None: break

                image = process_img(image_rgb)
                self.frame_stack.append(image)
                next_state = self._get_stacked_state()
                done = 1 if collision else 0
                
                self.total_rewards += reward
                episode_reward += reward
                
                replay_buffer.add(state, action, next_state, reward, done)

                loss_val = 0
                if not eval:
                    if ep > self.start_train and (self.global_t % self.train_freq) == 0:
                        loss_val = model.train(replay_buffer)
                        self.logger.log_loss(loss_val)

                # Live Terminal Update
                print(f"\rEp: {ep} | Step: {counter}/{self.max_iter} | Speed: {int(speed)} km/h | Rew: {reward:.2f} | Loss: {loss_val:.4f}", end='')

                if collision == 1 or counter >= self.max_iter or dist > self.max_dist_from_waypoint:
                    break
            
            avg_speed = episode_speed_sum / counter if counter > 0 else 0
            distance_traveled = start_location.distance(self.vehicle.get_location())
            
            print(f"\nEpisode {ep} Done. Total Reward: {episode_reward:.2f}, Dist: {distance_traveled:.1f}m, Crashed: {bool(collision)}")
            
            self.logger.log_episode(episode_reward, counter, avg_speed, model.current_eps, distance_traveled, 1 if collision else 0)
            self.logger.save_training_plots()
            self.logger.save_episode_plots(ep)

            if ep % self.save_freq == 0 and ep > 0:
                self.save(model, ep)

    def save(self, model, ep):
        if ep % self.save_freq == 0 and ep > self.start_ep:
            avg_reward = self.total_rewards/self.save_freq
            self.average_rewards_list.append(avg_reward)
            self.total_rewards = 0
            model.save('weights/model_ep_{}'.format(ep))
            print("Saved model weights.")
    
    def quit(self):
        pass

def get_reward_comp(vehicle, waypoint, collision):
    vehicle_location = vehicle.get_location()
    x_wp = waypoint.transform.location.x
    y_wp = waypoint.transform.location.y
    x_vh = vehicle_location.x
    y_vh = vehicle_location.y
    wp_array = np.array([x_wp, y_wp])
    vh_array = np.array([x_vh, y_vh])
    dist = np.linalg.norm(wp_array - vh_array)
    vh_yaw = correct_yaw(vehicle.get_transform().rotation.yaw)
    wp_yaw = correct_yaw(waypoint.transform.rotation.yaw)
    cos_yaw_diff = np.cos((vh_yaw - wp_yaw)*np.pi/180.)
    collision = 0 if collision is None else 1
    return cos_yaw_diff, dist, collision

def reward_value(cos_yaw_diff, dist, collision, lambda_1=1, lambda_2=1, lambda_3=5):
    reward = (lambda_1 * cos_yaw_diff) - (lambda_2 * dist) - (lambda_3 * collision)
    return reward