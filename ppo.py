import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from rl_env import TetrisEnv
from game_utils import *

BOARD_W, BOARD_H = 5, 10

RUN_ID = time.strftime("%Y%m%d-%H%M%S")
RUN_DIR = os.path.join("runs", f"ppo_tetris_{BOARD_W}x{BOARD_H}_{RUN_ID}")
MODEL_PATH = os.path.join(RUN_DIR, "ppo_final.pth")


def get_state(env):
    return env.game.get_state_matrix().astype(np.float32)


class ActorCritic(nn.Module):
    def __init__(self, board_height, board_width, n_actions, in_channels):
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

        self.policy_head = nn.Sequential(
            nn.Linear(conv_out, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

        self.value_head = nn.Sequential(
            nn.Linear(conv_out, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value


def train_ppo(
    num_updates=2000,
    steps_per_update=4096,
    minibatch_size=256,
    epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_ratio=0.2,
    lr=3e-4,
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    save_interval=100,
):
    os.makedirs(RUN_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TetrisEnv(width=BOARD_W, height=BOARD_H, render_mode=None)

    env.reset()
    s = get_state(env)
    in_channels = s.shape[0]
    n_actions = env.action_space.n

    hparams = {
        "board_w": BOARD_W,
        "board_h": BOARD_H,
        "num_updates": num_updates,
        "steps_per_update": steps_per_update,
        "minibatch_size": minibatch_size,
        "epochs": epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_ratio": clip_ratio,
        "lr": lr,
        "vf_coef": vf_coef,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm,
        "save_interval": save_interval,
        "device": str(device),
    }
    with open(os.path.join(RUN_DIR, "hparams.json"), "w") as f:
        json.dump(hparams, f, indent=2)

    model = ActorCritic(BOARD_H, BOARD_W, n_actions, in_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ep_returns = []
    ep_scores = []
    ep_lines = []
    best_score = 0.0

    print("Training PPO...")
    print(f"Device:       {device}")
    print(f"Board:        {BOARD_W} x {BOARD_H}")
    print(f"Obs shape:    {s.shape} (C,H,W)")
    print(f"Actions:      {n_actions}")
    print(f"Run dir:      {RUN_DIR}")

    global_step = 0
    cur_obs = s
    cur_done = False

    for update in range(1, num_updates + 1):
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []

        steps_collected = 0

        while steps_collected < steps_per_update:
            if cur_done:
                env.reset()
                cur_obs = get_state(env)
                cur_done = False

            obs_buf.append(cur_obs)
            obs_tensor = torch.from_numpy(cur_obs).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, value = model(obs_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)

            a = int(action.item())
            _, reward, terminated, truncated, info = env.step(a)
            cur_done = terminated or truncated
            next_obs = get_state(env)

            act_buf.append(a)
            logp_buf.append(float(logprob.cpu().item()))
            rew_buf.append(float(reward))
            done_buf.append(float(cur_done))
            val_buf.append(float(value.cpu().item()))

            cur_obs = next_obs
            global_step += 1
            steps_collected += 1

            if cur_done:
                ep_returns.append(sum(rew_buf[-steps_collected:]))
                ep_scores.append(info.get("score", 0.0))
                ep_lines.append(info.get("lines_cleared", 0.0))
                best_score = max(best_score, ep_scores[-1])

        obs_buf = np.array(obs_buf, dtype=np.float32)  # (T,C,H,W)
        act_buf = np.array(act_buf, dtype=np.int64)    # (T,)
        logp_buf = np.array(logp_buf, dtype=np.float32)
        rew_buf = np.array(rew_buf, dtype=np.float32)
        done_buf = np.array(done_buf, dtype=np.float32)
        val_buf = np.array(val_buf, dtype=np.float32)

        with torch.no_grad():
            obs_tensor = torch.from_numpy(cur_obs).unsqueeze(0).to(device)
            _, next_value = model(obs_tensor)
            next_value = float(next_value.cpu().item())

        T = steps_per_update
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_nonterminal = 1.0 - done_buf[t]
                next_val = next_value
            else:
                next_nonterminal = 1.0 - done_buf[t + 1]
                next_val = val_buf[t + 1]
            delta = rew_buf[t] + gamma * next_val * next_nonterminal - val_buf[t]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + val_buf

        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        b_obs = torch.from_numpy(obs_buf).to(device)
        b_act = torch.from_numpy(act_buf).to(device)
        b_logp_old = torch.from_numpy(logp_buf).to(device)
        b_adv = torch.from_numpy(advantages).to(device)
        b_ret = torch.from_numpy(returns).to(device)
        b_val_old = torch.from_numpy(val_buf).to(device)

        batch_size = T
        idxs = np.arange(batch_size)

        clipfracs = []
        pi_losses = []
        v_losses = []
        entropies = []

        for _ in range(epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = idxs[start:start + minibatch_size]

                mb_obs = b_obs[mb_idx]
                mb_act = b_act[mb_idx]
                mb_logp_old = b_logp_old[mb_idx]
                mb_adv = b_adv[mb_idx]
                mb_ret = b_ret[mb_idx]

                logits, value = model(mb_obs)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - mb_logp_old)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                v_loss = (value - mb_ret).pow(2).mean()

                loss = policy_loss + vf_coef * v_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                clipfracs.append(
                    (torch.abs(ratio - 1.0) > clip_ratio).float().mean().item()
                )
                pi_losses.append(policy_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy.item())

        if ep_returns:
            recent = min(100, len(ep_returns))
            avg_ret = float(np.mean(ep_returns[-recent:]))
            avg_score = float(np.mean(ep_scores[-recent:]))
            avg_lines = float(np.mean(ep_lines[-recent:]))
        else:
            avg_ret = avg_score = avg_lines = 0.0

        approx_kl = float((b_logp_old - b_logp_old).mean().item())  # placeholder
        clipfrac = float(np.mean(clipfracs)) if clipfracs else 0.0
        pi_loss_mean = float(np.mean(pi_losses)) if pi_losses else 0.0
        v_loss_mean = float(np.mean(v_losses)) if v_losses else 0.0
        ent_mean = float(np.mean(entropies)) if entropies else 0.0

        if update % 10 == 0:
            print(
                f"Upd {update:4d}/{num_updates} | "
                f"Steps: {global_step:7d} | "
                f"Ret: {avg_ret:6.2f} | "
                f"Score: {avg_score:6.1f} | "
                f"Lines: {avg_lines:4.2f} | "
                f"Best: {best_score:6.1f} | "
                f"pi: {pi_loss_mean:6.3f} | "
                f"v: {v_loss_mean:6.3f} | "
                f"ent: {ent_mean:5.3f} | "
                f"clip: {clipfrac:4.2f}"
            )

        if update % save_interval == 0:
            ckpt_path = os.path.join(RUN_DIR, f"ppo_upd{update}.pth")
            torch.save(
                {
                    "update": update,
                    "model_state_dict": model.state_dict(),
                    "best_score": best_score,
                },
                ckpt_path,
            )
            print(f"  ✓ Saved checkpoint: {ckpt_path}")

    torch.save(
        {
            "update": num_updates,
            "model_state_dict": model.state_dict(),
            "best_score": best_score,
        },
        MODEL_PATH,
    )

    metrics_path = os.path.join(RUN_DIR, "metrics.npz")
    np.savez(
        metrics_path,
        returns=np.array(ep_returns, dtype=np.float32),
        scores=np.array(ep_scores, dtype=np.float32),
        lines=np.array(ep_lines, dtype=np.float32),
    )

    print("Training completed.")
    print(f"Final model saved to:   {MODEL_PATH}")
    print(f"Metrics saved to:       {metrics_path}")
    print(f"Hyperparams saved to:   {os.path.join(RUN_DIR, 'hparams.json')}")
    env.close()
    return model


def eval_policy(model_path, num_episodes=5, render=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    render_mode = "human" if render else None
    env = TetrisEnv(width=BOARD_W, height=BOARD_H, render_mode=render_mode)

    env.reset()
    s = get_state(env)
    in_channels = s.shape[0]
    n_actions = env.action_space.n

    model = ActorCritic(BOARD_H, BOARD_W, n_actions, in_channels).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from: {model_path}")
    print(f"Best score during training: {checkpoint.get('best_score', 'unknown')}")
    print(f"\nEvaluating for {num_episodes} episodes...\n")

    scores = []
    lines_list = []

    for ep in range(1, num_episodes + 1):
        env.reset()
        obs = get_state(env)
        done = False

        while not done:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                logits, _ = model(obs_t)
                action = torch.argmax(logits, dim=-1)
                a = int(action.item())

            _, _, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            obs = get_state(env)

            if render and env.render_mode == "human":
                env.game.render()
                time.sleep(0.05)

        scores.append(info.get("score", 0))
        lines_list.append(info.get("lines_cleared", 0))
        print(f"Episode {ep}: Score={scores[-1]}, Lines={lines_list[-1]}")

    env.close()


if __name__ == "__main__":
    model = train_ppo()
    print("Evaluating trained model...")
    eval_policy(MODEL_PATH, num_episodes=3, render=True)
