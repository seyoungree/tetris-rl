import os
import time
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rl_env import TetrisEnv
from game_utils import *

BOARD_W, BOARD_H = 8, 16
RUN_ID = time.strftime('%Y%m%d-%H%M%S')
RUN_DIR = os.path.join("runs", f"vector_dqn_{BOARD_W}x{BOARD_H}_{RUN_ID}")
MODEL_PATH = os.path.join(RUN_DIR, "dqn_final.pth")

class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def push(self, s, a, r, s2, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.priorities.append(max_priority)
        
        self.buffer[self.pos] = (s, a, r, s2, done)
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(self.priorities[:len(self.buffer)])
        
        # Compute sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


class VectorDQN(nn.Module):
	def __init__(self, state_dim, n_actions):
		super().__init__()
		
		self.network = nn.Sequential(
			nn.Linear(state_dim, 256),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, n_actions)
		)
	
	def forward(self, x):
		return self.network(x)


def get_state(env):
	return env.game.get_state_vector()


def train_vector_dqn(
	num_episodes=100000,
	buffer_capacity=70000,
	batch_size=128,
	gamma=0.99,
	lr=2e-4,
	start_learning=5000,
	target_update_interval=1000,
	epsilon_start=1.0,
	epsilon_end=0.04,
	epsilon_decay_steps=70000,
	save_interval=10000,
	update_freq=1,
):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	env = TetrisEnv(width=BOARD_W, height=BOARD_H, render_mode=None)
	env.reset()
	s = get_state(env)
	state_dim = len(s)
	n_actions = env.action_space.n

	os.makedirs(RUN_DIR, exist_ok=True)
	hparams = {
		"board_w": BOARD_W,
		"board_h": BOARD_H,
		"state_dim": state_dim,
		"num_episodes": num_episodes,
		"buffer_capacity": buffer_capacity,
		"batch_size": batch_size,
		"gamma": gamma,
		"lr": lr,
		"start_learning": start_learning,
		"target_update_interval": target_update_interval,
		"epsilon_start": epsilon_start,
		"epsilon_end": epsilon_end,
		"epsilon_decay_steps": epsilon_decay_steps,
		"save_interval": save_interval,
		"update_freq": update_freq,
		"device": str(device),
	}
	with open(os.path.join(RUN_DIR, "hparams.json"), "w") as f:
		json.dump(hparams, f, indent=2)
		
	print("Training Vector DQN...")
	print(f"State dim: {state_dim}, action dim: {n_actions}")
	print(f"Using device: {device}")
	print(f"model dir: {RUN_DIR}")
	q_net = VectorDQN(state_dim, n_actions).to(device)
	target_net = VectorDQN(state_dim, n_actions).to(device)
	target_net.load_state_dict(q_net.state_dict())
	target_net.eval()

	optimizer = optim.AdamW(q_net.parameters(), lr=lr, weight_decay=1e-5)
	replay = ReplayBuffer(buffer_capacity)

	epsilon = epsilon_start
	step_count = 0
	training_updates = 0

	episode_rewards = []
	episode_scores = []
	episode_lines = []
	best_score = 0
	best_lines = 0
	
	start_time = time.time()

	for ep in range(1, num_episodes + 1):
		env.reset()
		s = get_state(env)
		done = False

		ep_reward = 0.0

		while not done:
			step_count += 1

			if random.random() < epsilon:
				a = random.randrange(n_actions)
			else:
				with torch.no_grad():
					t = torch.from_numpy(s).unsqueeze(0).to(device)
					q_vals = q_net(t)
					a = int(q_vals.argmax(1).item())

			_, reward, terminated, truncated, info = env.step(a)
			done = terminated or truncated

			s2 = get_state(env)
			ep_reward += reward
			replay.push(s, a, reward, s2, float(done))
			s = s2

			# Decay epsilon
			if step_count <= epsilon_decay_steps:
				frac = step_count / epsilon_decay_steps
				epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)
			else:
				epsilon = epsilon_end

			# Train only every update_freq steps
			if len(replay) >= max(start_learning, batch_size) and step_count % update_freq == 0:
				training_updates += 1
				beta = min(1.0, 0.4 + 0.6 * step_count / num_episodes)
				batch, indices, weights = replay.sample(batch_size, beta)
				weights = torch.tensor(weights, dtype=torch.float32, device=device)
				states, actions, rewards, next_states, dones = zip(*batch)

				states = torch.from_numpy(np.stack(states)).to(device)
				next_states = torch.from_numpy(np.stack(next_states)).to(device)
				actions = torch.tensor(actions, dtype=torch.long, device=device)
				rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
				dones = torch.tensor(dones, dtype=torch.float32, device=device)

				q_values = q_net(states)
				q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

				# Double DQN
				with torch.no_grad():
					next_q_online = q_net(next_states)
					next_actions = next_q_online.argmax(1)
					next_q_target = target_net(next_states)
					next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
					target = rewards + gamma * next_q * (1.0 - dones)

				td_errors = torch.abs(q - target).detach().cpu().numpy()
				loss = (weights * nn.SmoothL1Loss(reduction='none')(q, target)).mean()

				optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
				optimizer.step()
				
				replay.update_priorities(indices, td_errors + 1e-6)
				if training_updates % target_update_interval == 0:
					target_net.load_state_dict(q_net.state_dict())

		final_score = info.get("score", 0)
		final_lines = info.get("lines_cleared", 0)

		episode_rewards.append(ep_reward)
		episode_scores.append(final_score)
		episode_lines.append(final_lines)
		best_score = max(best_score, final_score)
		best_lines = max(best_lines, final_lines)

		if ep % 500 == 0:
			recent = min(500, len(episode_rewards))
			avg_r = np.mean(episode_rewards[-recent:])
			avg_s = np.mean(episode_scores[-recent:])
			avg_l = np.mean(episode_lines[-recent:])
			elapsed = time.time() - start_time
			eps_per_sec = ep / elapsed
			eta_minutes = (num_episodes - ep) / eps_per_sec / 60
			
			print(
				f"Ep {ep:6d}/{num_episodes} | "
				f"Lines: {avg_l:5.2f} | "
				f"Best: {best_lines:4d} | "
				f"Reward: {avg_r:6.1f} | "
				f"ε: {epsilon:.3f} | "
				f"Speed: {eps_per_sec:.1f} ep/s | "
				f"ETA: {eta_minutes:.0f}m"
			)

		if ep % save_interval == 0:
			ckpt_path = os.path.join(RUN_DIR, f"dqn_ep{ep}.pth")
			torch.save({"episode": ep,
			   			"model_state_dict": q_net.state_dict(),
						"optimizer_state_dict": optimizer.state_dict(),
						"epsilon": epsilon,
						"best_score": best_score,
						"best_lines": best_lines,
				},
				ckpt_path,
			)
			print(f"Checkpoint saved to: {ckpt_path}")
	torch.save({
			"episode": num_episodes,
			"model_state_dict": q_net.state_dict(),
			"best_score": best_score,
			"best_lines": best_lines,
		},
		MODEL_PATH,
	)

	metrics_path = os.path.join(RUN_DIR, "metrics.npz")
	np.savez(
		metrics_path,
		rewards=np.array(episode_rewards, dtype=np.float32),
		scores=np.array(episode_scores, dtype=np.float32),
		lines=np.array(episode_lines, dtype=np.float32),
	)
	print(f"Model saved to: {MODEL_PATH}")
	env.close()
	return q_net


def eval_greedy(model_path, num_episodes=10, render=False):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	render_mode = "human" if render else "rgb_array"
	env = TetrisEnv(width=BOARD_W, height=BOARD_H, render_mode=render_mode)
	env.reset()
	s = get_state(env)
	state_dim = len(s)
	n_actions = env.action_space.n

	q_net = VectorDQN(state_dim, n_actions).to(device)
	checkpoint = torch.load(model_path, map_location=device)
	q_net.load_state_dict(checkpoint["model_state_dict"])
	q_net.eval()

	print(f"Loaded model from: {model_path}")
	print(f"\nEvaluating {num_episodes} episodes...\n")

	scores = []
	lines_list = []

	for ep in range(1, num_episodes + 1):
		env.reset()
		s = get_state(env)
		done = False

		while not done:
			with torch.no_grad():
				t = torch.from_numpy(s).unsqueeze(0).to(device)
				a = int(q_net(t).argmax(1).item())

			_, _, terminated, truncated, info = env.step(a)
			done = terminated or truncated
			s = get_state(env)

			if render:
				env.game.render()
				time.sleep(0.05)

		scores.append(info.get("score", 0))
		lines_list.append(info.get("lines_cleared", 0))
		print(f"Ep {ep}: Score={scores[-1]:4d}, Lines={lines_list[-1]:3d}")

if __name__ == "__main__":
	q_net = train_vector_dqn()
	print("Evaluating model...")
	eval_greedy(MODEL_PATH, num_episodes=10, render=False)