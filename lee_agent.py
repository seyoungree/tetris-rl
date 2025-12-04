# lee_ai.py
import time
import numpy as np
from game import TetrisGame
from rl_env import TetrisEnv

# -------------------------
# Lee Yiyuan Official Weights
# -------------------------
W_HEIGHT    = -0.510066
W_LINES     =  0.760666
W_HOLES     = -0.35663
W_BUMPINESS = -0.184483


# -------------------------
# Helper functions
# -------------------------

def clone_game_state(game):
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

def get_column_heights(board):
    h, w = board.shape
    heights = []
    for col in range(w):
        col_cells = board[:, col]
        filled_idxs = np.where(col_cells != 0)[0]
        if filled_idxs.size == 0:
            heights.append(0)
        else:
            top_filled = filled_idxs[0]      # smallest row index with a block
            heights.append(h - top_filled)   # height measured from bottom
    return heights


def count_holes(board):
    h, w = board.shape
    holes = 0
    for col in range(w):
        col_cells = board[:, col]
        filled = False
        for row in range(h):
            if col_cells[row] != 0:
                filled = True
            elif filled:
                holes += 1
    return holes


def bumpiness(heights):
    return sum(abs(heights[i] - heights[i+1])
               for i in range(len(heights)-1))


# -----------------------------------------
# Evaluation Function (Lee’s)
# -----------------------------------------

def evaluate_board(sim, prev_lines):
    board = sim.board

    heights = get_column_heights(board)
    agg_height = sum(heights)

    holes = count_holes(board)

    bumps = bumpiness(heights)

    cleared = sim.lines_cleared - prev_lines

    score = (
        W_HEIGHT    * agg_height +
        W_LINES     * cleared +
        W_HOLES     * holes +
        W_BUMPINESS * bumps
    )
    return score


# -----------------------------------------
# Action Selection
# -----------------------------------------

def choose_action(env):
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

        # rotate
        for _ in range(rot % 4):
            sim.rotate_piece()

        # move horizontally
        while sim.current_pos[1] < col:
            if not sim.move_piece(1): break
        while sim.current_pos[1] > col:
            if not sim.move_piece(-1): break

        sim.hard_drop()

        if sim.game_over:
            score = -1e9
        else:
            score = evaluate_board(sim, prev_lines)

        if best_score is None or score > best_score:
            best_score = score
            best_action = action

    return best_action, best_score


# -----------------------------------------
# Episode Runner
# -----------------------------------------

def run_episode(width=10, height=20, render=True, seed=None):
    mode = "human" if render else None
    env = TetrisEnv(width=width, height=height, render_mode=mode, seed=seed)

    obs, info = env.reset()
    done = False

    while not done:
        act, _ = choose_action(env)
        obs, reward, terminated, truncated, info = env.step(act)
        done = terminated or truncated

        if render:
            env.game.render()
            time.sleep(0.03)

    env.close()
    print(f"Finished: score={info['score']} lines={info['lines_cleared']}")
    return info['lines_cleared']


if __name__ == "__main__":
    results = []
    for _ in range(5):
        results.append(run_episode(width=10, height=20, render=False, seed=0))

    print("Mean:", np.mean(results))
    print("Std:", np.std(results))
