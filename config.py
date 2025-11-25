# config.py

# Discrete actions: Steer [-0.5 to 0.5]
action_values = [-0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5]
action_map = {i:x for i, x in enumerate(action_values)}

env_params = {
    'target_speed': 10,        
    'max_iter': 2000,          
    
    # --- CHANGE IS HERE ---
    'start_buffer': 2000,       # WAS 2000. Reduced to 200 so training starts in Episode 0.
    # ----------------------
    
    'train_freq': 4,           
    'save_freq': 100,
    'start_ep': 0,
    'max_dist_from_waypoint': 20,
    'lambda_collision': 200    
}

# Hyperparameters
buffer_size = 50000            
batch_size = 128               
lr = 1e-4                      
gamma = 0.99                   
epsilon_decay = 100000