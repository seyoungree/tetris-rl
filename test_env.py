import time, argparse
from rl_env import TetrisEnv

def run_test(args):
	print(f"\n[TETRIS TEST] render={args.render}, episodes={args.episodes}, delay={args.delay}\n")

	render_mode = None if args.render == "none" else args.render
	env = TetrisEnv(width=10, height=20, render_mode=render_mode)

	for ep in range(1, args.episodes + 1):
		obs, info = env.reset()
		print(f"Episode {ep}/{args.episodes} | lines_start={info.get('lines_cleared',0)}")
		done = False
		step = 0

		while not done:
			action = env.action_space.sample()
			obs, reward, terminated, truncated, info = env.step(action)
			done = terminated or truncated

			print(f"  step {step}: a={action}, r={reward:.2f}, lc={info.get('lines_cleared_move',0)}, h={info.get('height',0)}, holes={info.get('holes',0)}")

			if render_mode == "human":
				env.game.render()
			elif render_mode == "rgb_array":
				env.game.render()
			
			time.sleep(args.delay)
			step += 1

		print(f"Episode {ep} ended | total_lines={info.get('lines_cleared',0)}\n")

	env.close()


if __name__ == "__main__":
	p = argparse.ArgumentParser()
	p.add_argument("--render", choices=["human","rgb_array","none"], default="human")
	p.add_argument("--episodes", type=int, default=5)
	p.add_argument("--delay", type=float, default=0.2)
	args = p.parse_args()

	run_test(args)
