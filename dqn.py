import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rl_env import TetrisEnv

BOARD_W, BOARD_H = 10, 20
MODEL_PATH = f"models/dqn_tetris_{BOARD_W}x{BOARD_H}.pth"

class ReplayBuffer:
    def __init__(self, capacity, rng):
        self.capacity, self.rng = capacity, rng
        self.buffer, self.pos = [], 0

    def push(self, s, a, r, s2, done):
        data = (s, a, r, s2, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        idxs = self.rng.integers(len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in idxs]
        s, a, r, s2, d = zip(*batch)
        return (
            np.stack(s),
            np.array(a, np.int64),
            np.array(r, np.float32),
            np.stack(s2),
            np.array(d, np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def get_state(env):
    mat = env.game.get_state_matrix().astype(np.float32)  # -1 (current), 0, 1..7
    mat = np.clip(mat, -1, 7) / 7.0                       # normalize to ~[-0.14, 1]
    return mat.reshape(-1)


def train_dqn(
    num_episodes=2000,
    buffer_capacity=50_000,
    batch_size=64,
    gamma=0.99,
    lr=1e-3,
    start_learning=1_000,
    target_update_interval=1_000,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_steps=200_000,
):
    rng = np.random.default_rng()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = TetrisEnv(width=BOARD_W, height=BOARD_H, render_mode="rgb_array")
    env.reset()
    s = get_state(env)
    input_dim = s.size
    n_actions = env.action_space.n

    print(f"Device: {device} | Board: {BOARD_W}x{BOARD_H} | State dim: {input_dim} | Actions: {n_actions}")

    q_net = DQN(input_dim, n_actions).to(device)
    target_net = DQN(input_dim, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    opt = optim.Adam(q_net.parameters(), lr=lr)
    replay = ReplayBuffer(buffer_capacity, rng)

    epsilon = epsilon_start
    step_count = 0

    for ep in range(1, num_episodes + 1):
        env.reset()
        s = get_state(env)
        done = False
        ep_reward = 0.0
        total_lines = 0

        while not done:
            step_count += 1

            # ε-greedy
            if rng.random() < epsilon:
                a = rng.integers(n_actions)
            else:
                with torch.no_grad():
                    t = torch.from_numpy(s).unsqueeze(0).to(device)
                    a = int(q_net(t).argmax(dim=1).item())

            _, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            s2 = get_state(env)

            ep_reward += r
            total_lines = info.get("lines_cleared", total_lines)

            replay.push(s, a, r, s2, done)
            s = s2

            # epsilon decay
            epsilon = max(
                epsilon_end,
                epsilon_start - step_count / epsilon_decay_steps * (epsilon_start - epsilon_end),
            )

            # train
            if len(replay) >= start_learning:
                states_b, actions_b, rewards_b, next_states_b, dones_b = replay.sample(batch_size)

                states_b = torch.from_numpy(states_b).to(device)
                actions_b = torch.from_numpy(actions_b).to(device)
                rewards_b = torch.from_numpy(rewards_b).to(device)
                next_states_b = torch.from_numpy(next_states_b).to(device)
                dones_b = torch.from_numpy(dones_b).to(device)

                q = q_net(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q = target_net(next_states_b).max(dim=1)[0]
                    target = rewards_b + gamma * next_q * (1.0 - dones_b)

                loss = nn.MSELoss()(q, target)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                opt.step()

                if step_count % target_update_interval == 0:
                    target_net.load_state_dict(q_net.state_dict())

        if ep % 50 == 0:
            print(f"Ep {ep}/{num_episodes} | Reward={ep_reward:.1f} | Lines={total_lines} | ε={epsilon:.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH) or "models", exist_ok=True)
    torch.save(q_net.state_dict(), MODEL_PATH)
    print(f"\nTraining done. Model saved to {MODEL_PATH}")
    env.close()
    return q_net


def eval_greedy(q_net, num_episodes=5, render=True):
    device = next(q_net.parameters()).device
    env = TetrisEnv(width=BOARD_W, height=BOARD_H, render_mode="human" if render else "rgb_array")

    scores, lines_list = [], []

    for ep in range(1, num_episodes + 1):
        env.reset()
        s = get_state(env)
        done = False

        while not done:
            with torch.no_grad():
                t = torch.from_numpy(s).unsqueeze(0).to(device)
                a = int(q_net(t).argmax(dim=1).item())
            _, _, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            s = get_state(env)

            if render:
                env.game.render()
                time.sleep(0.05)

        scores.append(info.get("score", 0))
        lines_list.append(info.get("lines_cleared", 0))
        print(f"[Eval] Ep {ep}: Score={scores[-1]}, Lines={lines_list[-1]}")

    print(f"\nEval avg Score={np.mean(scores):.2f}, avg Lines={np.mean(lines_list):.2f}")
    env.close()


if __name__ == "__main__":
    q_net = train_dqn(num_episodes=2_000)
    eval_greedy(q_net, num_episodes=3, render=True)