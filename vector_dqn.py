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

BOARD_W, BOARD_H = 10, 20
RUN_ID = time.strftime('%Y%m%d-%H%M%S')
RUN_DIR = os.path.join("runs", f"vector_dqn_{BOARD_W}x{BOARD_H}_{RUN_ID}")
MODEL_PATH = os.path.join(RUN_DIR, "dqn_final.pth")
MODEL_PATH = "/Users/seyoungree/tetris-rl/runs/vector_dqn_10x20_20251203-090921/dqn_ep10000.pth"
class ReplayBuffer:
	"""
	Prioritized Experience Replay buffer.
	Stores transitions (s, a, r, s', done) and priorities for sampling.
	"""

	def __init__(self, capacity, alpha=0.6):
		self.capacity = capacity
		self.alpha = alpha

		self.buffer = [None] * capacity
		self.priorities = np.zeros((capacity,), dtype=np.float32)

		self.pos = 0
		self.size = 0

	def push(self, s, a, r, s2, done):
		"""Add a new transition with max priority so it is likely to be sampled soon."""
		max_prio = self.priorities.max() if self.size > 0 else 1.0

		self.buffer[self.pos] = (s, a, r, s2, done)
		self.priorities[self.pos] = max_prio

		self.pos = (self.pos + 1) % self.capacity
		self.size = min(self.size + 1, self.capacity)

	def sample(self, batch_size, beta=0.4):
		assert self.size > 0, "Cannot sample from an empty buffer"

		prios = self.priorities[:self.size]
		# avoid zero probabilities
		prios = np.where(prios > 0, prios, 1e-6)

		probs = prios ** self.alpha
		probs /= probs.sum()

		indices = np.random.choice(self.size, batch_size, p=probs)
		samples = [self.buffer[idx] for idx in indices]

		total = self.size
		weights = (total * probs[indices]) ** (-beta)
		weights /= weights.max()  # normalize for stability

		return samples, indices, weights

	def update_priorities(self, indices, new_priorities, max_priority=10.0):
		"""Update priorities for a set of transitions."""
		new_priorities = np.asarray(new_priorities, dtype=np.float32)
		# small epsilon, clip to avoid exploding priorities
		new_priorities = np.clip(new_priorities, 1e-6, max_priority)
		for idx, prio in zip(indices, new_priorities):
			self.priorities[idx] = prio

	def __len__(self):
		return self.size


class VectorDQN(nn.Module):
	def __init__(self, state_dim, n_actions):
		super().__init__()

		self.network = nn.Sequential(
			nn.Linear(state_dim, 512),
			nn.ReLU(),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, n_actions),
		)

	def forward(self, x):
		return self.network(x)


def get_state(env):
	# Ensure get_state_vector returns a 1D float32 numpy array
	s = env.game.get_state_vector()
	return np.asarray(s, dtype=np.float32)


def train_vector_dqn(
	num_episodes=100000,
	buffer_capacity=70000,
	batch_size=128,
	gamma=0.99,
	lr=2e-4,
	start_learning=5000,
	target_update_interval=1000,   # in number of gradient updates
	epsilon_start=1.0,
	epsilon_end=0.04,
	epsilon_decay_steps=70000,     # in environment steps
	save_interval=10000,
	update_freq=1,                 # update every N steps
	per_alpha=0.6,
	per_beta_start=0.4,
	per_beta_end=1.0,
	per_beta_frames=200000,        # how many training updates until beta ~ 1.0
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
		"per_alpha": per_alpha,
		"per_beta_start": per_beta_start,
		"per_beta_end": per_beta_end,
		"per_beta_frames": per_beta_frames,
	}
	with open(os.path.join(RUN_DIR, "hparams.json"), "w") as f:
		json.dump(hparams, f, indent=2)

	print("Training Vector DQN...")
	print(f"State dim: {state_dim}, action dim: {n_actions}")
	print(f"Using device: {device}")

	q_net = VectorDQN(state_dim, n_actions).to(device)
	target_net = VectorDQN(state_dim, n_actions).to(device)
	target_net.load_state_dict(q_net.state_dict())
	target_net.eval()

	optimizer = optim.AdamW(q_net.parameters(), lr=lr, weight_decay=1e-5)
	replay = ReplayBuffer(buffer_capacity, alpha=per_alpha)

	epsilon = epsilon_start
	step_count = 0          # environment steps
	training_updates = 0    # gradient steps

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

			# Epsilon-greedy policy
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

			# Linear epsilon decay over epsilon_decay_steps
			if step_count < epsilon_decay_steps:
				frac = step_count / float(epsilon_decay_steps)
				epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)
			else:
				epsilon = epsilon_end

			# Train the network
			if len(replay) >= max(start_learning, batch_size) and step_count % update_freq == 0:
				training_updates += 1

				# PER beta schedule (increase toward 1.0 over per_beta_frames updates)
				beta_frac = min(1.0, training_updates / float(per_beta_frames))
				beta = per_beta_start + beta_frac * (per_beta_end - per_beta_start)

				batch, indices, weights = replay.sample(batch_size, beta)

				states, actions, rewards, next_states, dones = zip(*batch)

				states = torch.from_numpy(np.stack(states)).to(device)
				next_states = torch.from_numpy(np.stack(next_states)).to(device)
				actions = torch.tensor(actions, dtype=torch.long, device=device)
				rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
				dones = torch.tensor(dones, dtype=torch.float32, device=device)
				weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

				# Q(s,a)
				q_values = q_net(states)
				q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

				# Double DQN target
				with torch.no_grad():
					next_q_online = q_net(next_states)
					next_actions = next_q_online.argmax(1)

					next_q_target = target_net(next_states)
					next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

					target = rewards + gamma * next_q * (1.0 - dones)

				td_errors = (q - target).detach().cpu().numpy()
				per_prios = np.abs(td_errors) + 1e-6

				# PER-weighted loss
				loss_per_sample = nn.SmoothL1Loss(reduction='none')(q, target)
				loss = (weights_t * loss_per_sample).mean()

				optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
				optimizer.step()

				# Update priorities
				replay.update_priorities(indices, per_prios)

				# Periodically update target network
				if training_updates % target_update_interval == 0:
					target_net.load_state_dict(q_net.state_dict())

		# End of episode logging
		final_score = info.get("score", 0)
		final_lines = info.get("lines_cleared", 0)

		episode_rewards.append(ep_reward)
		episode_scores.append(final_score)
		episode_lines.append(final_lines)
		best_score = max(best_score, final_score)
		best_lines = max(best_lines, final_lines)

		if ep % 100 == 0:
			recent = min(100, len(episode_rewards))
			avg_r = np.mean(episode_rewards[-recent:])
			avg_s = np.mean(episode_scores[-recent:])
			avg_l = np.mean(episode_lines[-recent:])
			elapsed = time.time() - start_time
			eps_per_sec = ep / elapsed
			eta_minutes = (num_episodes - ep) / max(eps_per_sec, 1e-6) / 60.0

			print(
				f"Ep {ep:6d}/{num_episodes} | "
				f"Lines: {avg_l:6.2f} | "
				f"Best: {best_lines:4d} | "
				f"Reward: {avg_r:7.1f} | "
				f"ε: {epsilon:.3f} | "
				f"Updates: {training_updates:7d} | "
				f"Speed: {eps_per_sec:5.2f} ep/s | "
				f"ETA: {eta_minutes:4.0f}m"
			)

		if ep % save_interval == 0:
			ckpt_path = os.path.join(RUN_DIR, f"dqn_ep{ep}.pth")
			torch.save(
				{
					"episode": ep,
					"model_state_dict": q_net.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"epsilon": epsilon,
					"best_score": best_score,
					"best_lines": best_lines,
				},
				ckpt_path,
			)
			print(f"Checkpoint saved to: {ckpt_path}")

	# Final save
	torch.save(
		{
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
		print(f"Ep {ep:3d}: Score={scores[-1]:4d}, Lines={lines_list[-1]:4d}")

	env.close()


if __name__ == "__main__":
	q_net = train_vector_dqn()
	print("Evaluating model...")
	eval_greedy(MODEL_PATH, num_episodes=10, render=False)
