import torch
import os
import glob
import re
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from DQN_Control.replay_buffer import ReplayBuffer
from DQN_Control.model import DQN
# Importing buffer_size and batch_size from the "High Efficiency" config update
from config import action_map, env_params, buffer_size, batch_size
from utils import *
from environment import SimEnv

def get_latest_checkpoint(weights_dir='weights'):
    """Finds the latest checkpoint file in the weights directory."""
    if not os.path.exists(weights_dir):
        return None, 0
    
    # Look for model files
    files = glob.glob(os.path.join(weights_dir, "model_ep_*.pth"))
    if not files:
        return None, 0
    
    # Extract episode number using Regex (finds the highest number)
    latest_file = max(files, key=lambda f: int(re.search(r'model_ep_(\d+)', f).group(1)))
    latest_ep = int(re.search(r'model_ep_(\d+)', latest_file).group(1))
    
    # Return filename without extension
    return latest_file.replace('.pth', ''), latest_ep

def run():
    env = None
    try:
        # --- HYPERPARAMETERS ---
        # Note: buffer_size should be 50000 (from config) due to uint8 optimization
        print(f"Initializing Memory: Buffer Size {buffer_size}...")
        
        state_dim = (4, 128, 128) # 4 Stacked frames, 128x128 resolution
        in_channels = 4
        total_episodes = 10000
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_actions = len(action_map)
        
        # --- INITIALIZATION ---
        replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)
        model = DQN(num_actions, state_dim, in_channels, device)

        # --- RESUME LOGIC ---
        start_ep = 0
        latest_checkpoint, latest_ep = get_latest_checkpoint()

        if latest_checkpoint:
            print(f"\n⚠️ Found checkpoint: {latest_checkpoint}")
            print(f"Resuming training from Episode {latest_ep}...")
            
            # 1. Load Neural Network Weights
            model.load(latest_checkpoint)
            
            # 2. Load Replay Memory (The "Experience")
            # We look for the corresponding buffer file
            buffer_path = latest_checkpoint.replace("model_ep", "buffer_ep") + ".npz"
            
            if os.path.exists(buffer_path):
                replay_buffer.load(buffer_path)
            else:
                print("Warning: Model loaded, but Replay Buffer file not found. Starting with empty memory.")
            
            start_ep = latest_ep + 1
        else:
            print("\nNo checkpoints found. Starting fresh training.")

        # --- ENVIRONMENT START ---
        print("Starting CARLA Environment...")
        
        # Update start_ep in params so logs/plots x-axis is correct
        env_params['start_ep'] = start_ep
        
        # visuals=False because we use the CARLA Spectator Camera (Window A)
        env = SimEnv(visuals=False, **env_params) 

        # --- TRAINING LOOP ---
        for ep in range(start_ep, total_episodes):
            # 1. Spawn Car & Sensors
            env.create_actors()
            
            # 2. Run Episode (Training happens inside here)
            env.generate_episode(model, replay_buffer, ep, action_map, eval=False)
            
            # 3. Cleanup Actors
            env.reset()
            
            # 4. Periodic Save of Replay Buffer
            # (Model is saved inside generate_episode, but Buffer is saved here)
            if ep % env_params['save_freq'] == 0 and ep > start_ep:
                buffer_save_path = f"weights/buffer_ep_{ep}.npz"
                replay_buffer.save(buffer_save_path)

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Ensure we close connections to prevent "Ghost Sensors" slowing down CARLA
        if env is not None:
            print("Closing environment and cleaning up...")
            env.quit()

if __name__ == "__main__":
    run()