import torch
import numpy as np

class ReplayBuffer(object):
    def __init__(self, state_dim, batch_size, buffer_size, device) -> None:
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device
        self.ptr = 0
        self.crt_size = 0

        # OPTIMIZATION: Store as uint8 (0-255) to save 4x RAM
        # We divide by 255 only when sampling for training
        self.state = np.zeros((self.max_size,) + state_dim, dtype=np.uint8)
        self.action = np.zeros((self.max_size, 1), dtype=np.int64)
        self.next_state = np.zeros((self.max_size,) + state_dim, dtype=np.uint8)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.done = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(self, state, action, next_state, reward, done):
        # Convert float (0.0-1.0) back to uint8 (0-255) for storage
        state_int = (state * 255).astype(np.uint8)
        next_state_int = (next_state * 255).astype(np.uint8)

        self.state[self.ptr] = state_int
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state_int
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)

    def save(self, filename):
        print(f"Saving buffer to {filename}...")
        np.savez_compressed(
            filename,
            state=self.state[:self.crt_size],
            action=self.action[:self.crt_size],
            next_state=self.next_state[:self.crt_size],
            reward=self.reward[:self.crt_size],
            done=self.done[:self.crt_size],
            ptr=self.ptr,
            crt_size=self.crt_size
        )

    def load(self, filename):
        print(f"Loading buffer from {filename}...")
        try:
            data = np.load(filename)
            limit = min(data['crt_size'], self.max_size)
            self.state[:limit] = data['state'][:limit]
            self.action[:limit] = data['action'][:limit]
            self.next_state[:limit] = data['next_state'][:limit]
            self.reward[:limit] = data['reward'][:limit]
            self.done[:limit] = data['done'][:limit]
            self.ptr = int(data['ptr']) % self.max_size
            self.crt_size = limit
            print("Buffer loaded successfully.")
        except Exception as e:
            print(f"Load failed: {e}")

    def sample(self):
        ind = np.random.randint(0, self.crt_size, size=self.batch_size)
        
        # ON-THE-FLY NORMALIZATION (uint8 -> float32 0..1)
        state_batch = torch.FloatTensor(self.state[ind]).to(self.device) / 255.0
        next_state_batch = torch.FloatTensor(self.next_state[ind]).to(self.device) / 255.0
        
        return (
            state_batch,
            torch.LongTensor(self.action[ind]).to(self.device),
            next_state_batch,
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )