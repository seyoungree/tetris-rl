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
from lin_agent import choose_action

BOARD_W, BOARD_H = 5, 10
RUN_ID = time.strftime('%Y%m%d-%H%M%S')
RUN_DIR = os.path.join("runs", f"dqn_tetris_{BOARD_W}x{BOARD_H}_{RUN_ID}")
MODEL_PATH = os.path.join(RUN_DIR, "dqn_final.pth")

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, s, a, r, s2, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (s, a, r, s2, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, board_height, board_width, n_actions, in_channels=2, feat_dim=4):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, board_height, board_width)
            conv_out = self.conv(dummy).view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(conv_out + feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, board, feat):
        x = self.conv(board)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, feat], dim=1)
        return self.fc(x)

def get_state(env):
    mat = env.game.get_state_matrix().astype(np.float32)  # (2,H,W)
    height = get_board_height(env.game.board)
    holes = count_holes(env.game.board)
    bump = get_bumpiness(env.game.board)
    lines = env.game.lines_cleared
    feat = np.array([lines, height, holes, bump], dtype=np.float32)
    return mat, feat


def train_dqn(
    num_episodes=50000,
    buffer_capacity=50000,
    batch_size=64,
    gamma=0.99,
    lr=1e-4,
    start_learning=1000,
    target_update_interval=1000,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_steps=20000,
    save_interval=5000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TetrisEnv(width=BOARD_W, height=BOARD_H, render_mode=None)
    env.reset()
    s_img, s_feat = get_state(env)
    in_channels = s_img.shape[0]
    n_actions = env.action_space.n
    
    os.makedirs(RUN_DIR, exist_ok=True)
    hparams = {
        "board_w": BOARD_W,
        "board_h": BOARD_H,
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
        "device": str(device),
        "in_channels": in_channels,
    }
    with open(os.path.join(RUN_DIR, "hparams.json"), "w") as f:
        json.dump(hparams, f, indent=2)

    print("Training DQN...")
    q_net = DQN(BOARD_H, BOARD_W, n_actions, in_channels=in_channels).to(device)
    target_net = DQN(BOARD_H, BOARD_W, n_actions, in_channels=in_channels).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay = ReplayBuffer(buffer_capacity)

    epsilon = epsilon_start
    step_count = 0

    episode_rewards = []
    episode_scores = []
    episode_lines = []
    best_score = 0

    for ep in range(1, num_episodes + 1):
        env.reset()
        s_img, s_feat = get_state(env)
        done = False

        ep_reward = 0.0
        ep_losses = []

        while not done:
            step_count += 1

            if random.random() < epsilon:
                a = random.randrange(n_actions)
            else:
                with torch.no_grad():
                    t_img = torch.from_numpy(s_img).unsqueeze(0).to(device)        # (1,C,H,W)
                    t_feat = torch.from_numpy(s_feat).unsqueeze(0).to(device)      # (1,F)
                    q_vals = q_net(t_img, t_feat)
                    a = int(q_vals.argmax(1).item())

            _, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            s2_img, s2_feat = get_state(env)
            ep_reward += reward
            replay.push((s_img, s_feat), a, reward, (s2_img, s2_feat), float(done))
            s_img, s_feat = s2_img, s2_feat

            frac = min(1.0, step_count / epsilon_decay_steps)
            epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

            if len(replay) >= max(start_learning, batch_size):
                batch = replay.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                state_imgs, state_feats = zip(*states)
                next_imgs, next_feats = zip(*next_states)

                state_imgs = torch.from_numpy(np.stack(state_imgs)).to(device)         # (B,C,H,W)
                state_feats = torch.from_numpy(np.stack(state_feats)).to(device)       # (B,F)
                next_imgs = torch.from_numpy(np.stack(next_imgs)).to(device)
                next_feats = torch.from_numpy(np.stack(next_feats)).to(device)

                actions = torch.tensor(actions, dtype=torch.long, device=device)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                dones = torch.tensor(dones, dtype=torch.float32, device=device)

                q_values = q_net(state_imgs, state_feats)
                q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q = target_net(next_imgs, next_feats).max(1)[0]
                    target = rewards + gamma * next_q * (1.0 - dones)

                loss = nn.SmoothL1Loss()(q, target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()

                ep_losses.append(loss.item())

                if step_count % target_update_interval == 0:
                    target_net.load_state_dict(q_net.state_dict())

        final_score = info.get("score", 0)
        final_lines = info.get("lines_cleared", 0)

        episode_rewards.append(ep_reward)
        episode_scores.append(final_score)
        episode_lines.append(final_lines)
        best_score = max(best_score, final_score)

        if ep % 100 == 0:
            recent = min(100, len(episode_rewards))
            avg_r = np.mean(episode_rewards[-recent:])
            avg_s = np.mean(episode_scores[-recent:])
            avg_l = np.mean(episode_lines[-recent:])
            avg_loss = np.mean(ep_losses) if ep_losses else 0.0
            print(
                f"Ep {ep:5d}/{num_episodes} | "
                f"R: {avg_r:7.1f} | "
                f"Score: {avg_s:6.1f} | "
                f"Lines: {avg_l:4.2f} | "
                f"Best: {best_score:6.1f} | "
                f"ε: {epsilon:.3f} | "
                f"Loss: {avg_loss:.3f} | "
                f"Buf: {len(replay):5d}"
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
                },
                ckpt_path,
            )
            print(f"  ✓ Saved checkpoint: {ckpt_path}")

    torch.save(
        {
            "episode": num_episodes,
            "model_state_dict": q_net.state_dict(),
            "best_score": best_score,
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

    print("Training completed.")
    print(f"Final model saved to:   {MODEL_PATH}")
    print(f"Metrics saved to:       {metrics_path}")
    print(f"Hyperparams saved to:   {os.path.join(RUN_DIR, 'hparams.json')}")
    env.close()
    return q_net


def eval_greedy(model_path, num_episodes=5, render=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    render_mode = "human" if render else "rgb_array"
    env = TetrisEnv(width=BOARD_W, height=BOARD_H, render_mode=render_mode)
    env.reset()
    s_img, s_feat = get_state(env)
    in_channels = s_img.shape[0]
    n_actions = env.action_space.n

    q_net = DQN(BOARD_H, BOARD_W, n_actions, in_channels=in_channels).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    q_net.load_state_dict(checkpoint["model_state_dict"])
    q_net.eval()

    print(f"Loaded model from: {model_path}")
    print(f"\nEvaluating for {num_episodes} episodes...\n")

    scores = []
    lines_list = []
 
    for ep in range(1, num_episodes + 1):
        env.reset()
        s = get_state(env)
        done = False

        while not done:
            with torch.no_grad():
                t_img = torch.from_numpy(s_img).unsqueeze(0).to(device)
                t_feat = torch.from_numpy(s_feat).unsqueeze(0).to(device)
                a = int(q_net(t_img, t_feat).argmax(1).item())

            _, _, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            s_img, s_feat = get_state(env)

            if render:
                env.game.render()
                time.sleep(0.05)

        scores.append(info.get("score", 0))
        lines_list.append(info.get("lines_cleared", 0))
        print(f"Episode {ep}: Score={scores[-1]}, Lines={lines_list[-1]}")

    env.close()


if __name__ == "__main__":
    # q_net = train_dqn()
    print("Evaluating trained model...")
    eval_greedy(MODEL_PATH, num_episodes=3, render=True)
