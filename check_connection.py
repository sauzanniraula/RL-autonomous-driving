import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def test():
    try:
        print("1. Attempting to connect to CARLA on localhost:2000...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0) # 5 seconds timeout
        
        print("2. Connection object created. Pinging server...")
        world = client.get_world()
        
        print(f"3. SUCCESS! Connected to map: {world.get_map().name}")
        print("You are ready to run train.py now.")
        
    except RuntimeError as e:
        print("\n!!! CONNECTION FAILED !!!")
        print(f"Error: {e}")
        print("\nSOLUTIONS:")
        print("1. Make sure 'CarlaUE4.exe' is running in a separate window.")
        print("2. Make sure Windows Firewall is not blocking 'CarlaUE4.exe'.")
        print("3. Try restarting the computer if the port is stuck.")

if __name__ == '__main__':
    test()