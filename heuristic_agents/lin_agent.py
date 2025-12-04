import time
import numpy as np
from game import TetrisGame
from rl_env import TetrisEnv
from game_utils import (
	count_holes, row_transitions, column_transitions,
	calc_wells, landing_height, eroded_cells
)

W_HOLES      = -4.0
W_WELLS      = -1.0
W_ROW_TRANS  = -1.0
W_COL_TRANS  = -1.0
W_HEIGHT     = -1.0
W_ERODED     = +1.0


def clone_game_state(game):
	"""Manual clone—no pygame surfaces."""
	sim = TetrisGame(
		width=game.width,
		height=game.height,
		block_size=game.block_size,
		render_mode=None
	)
	sim.board = game.board.copy()
	sim.current_shape_name = game.current_shape_name
	sim.current_shape = game.current_shape.copy()
	sim.current_pos = list(game.current_pos)
	sim.game_over = game.game_over
	sim.score = game.score
	sim.lines_cleared = game.lines_cleared
	sim.piece_queue = list(game.piece_queue)
	return sim


def evaluate_board(sim_game, prev_lines, piece_final_y, piece_shape):
	board = sim_game.board
	h, w = board.shape

	holes = count_holes(board)
	rtrans = row_transitions(board)
	ctrans = column_transitions(board)
	well_sum = calc_wells(board)

	# lines cleared from THIS placement
	cleared = sim_game.lines_cleared - prev_lines

	# landing height
	land_h = landing_height(board, piece_final_y, piece_shape.shape[0], h)

	# eroded cells = cleared_lines * number_of_piece_cells_in_cleared_rows
	piece_cells = piece_shape.sum()
	eroded = eroded_cells(cleared, piece_cells)

	score = (
		W_HOLES     * holes +
		W_WELLS     * well_sum +
		W_ROW_TRANS * rtrans +
		W_COL_TRANS * ctrans +
		W_HEIGHT    * land_h +
		W_ERODED    * eroded
	)

	return score


def choose_action(env: TetrisEnv):
	game = env.game
	width = env.width
	n_actions = env.action_space.n

	best_score = None
	best_action = 0
	prev_lines = game.lines_cleared

	for action in range(n_actions):
		rot = action // width
		col = action % width

		sim = clone_game_state(game)
		shape = sim.current_shape.copy()

		# rotations
		for _ in range(rot % 4):
			sim.rotate_piece()

		# move horizontally
		while sim.current_pos[1] < col:
			if not sim.move_piece(1): break
		while sim.current_pos[1] > col:
			if not sim.move_piece(-1): break

		# final row BEFORE hard drop
		piece_final_y = sim.current_pos[0]

		# hard drop
		sim.hard_drop()
		if sim.game_over and sim.lines_cleared == prev_lines:
			score = -1e9
		else:
			score = evaluate_board(sim, prev_lines, piece_final_y, sim.current_shape)

		if (best_score is None) or (score > best_score):
			best_score = score
			best_action = action

	return best_action, best_score



def run_episodes(width=10, height=20, render=True, seed=None, num_episodes=10):
	mode = "human" if render else None
	env = TetrisEnv(width=width, height=height, render_mode=mode, seed=seed)
	results = []

	for i in range(1,num_episodes+1):
		obs, info = env.reset()
		done = False

		while not done:
			act, _ = choose_action(env)
			obs, reward, terminated, truncated, info = env.step(act)
			done = terminated or truncated
			
			if render:
				env.game.render()
				time.sleep(0.03)
		print(f"Finished: score={info['score']} lines={info['lines_cleared']}")
		results.append(info['lines_cleared'])

	env.close()
	print("Mean:", np.mean(results))
	print("Std:", np.std(results))

	

if __name__ == "__main__":
	run_episodes(width=10, height=20, render=False, seed=1)
