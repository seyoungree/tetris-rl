import numpy as np
import gymnasium as gym
from gymnasium import spaces
from game import TetrisGame
from game_utils import *
import time

class TetrisEnv(gym.Env):
    def __init__(self, width=10, height=20, render_mode="rgb_array"):
        super().__init__()
        self.width, self.height = width, height
        self.render_mode = render_mode
        self.game = TetrisGame(width=width, height=height, render_mode=render_mode)
        h, w, c = self.game.reset().shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(h,w,c), dtype=np.uint8)
        self.action_space = spaces.Discrete(4 * self.width)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        frame = self.game.reset()
        return frame, {}

    def _animate(self):
        if self.render_mode == "human":
            self.game.render()
            time.sleep(0.1)

    def step(self, action):
        if self.game.game_over:
            frame = self.game.render()
            return frame, 0.0, True, False, {}


        num_rotations = int((action // self.width) % 4)
        target_col = action % self.width

        prev_score = self.game.score
        prev_lines = self.game.lines_cleared

        # rotations
        for _ in range(num_rotations):
            self.game.rotate_piece()
            self._animate()

        # move horizontally
        while self.game.current_pos[1] < target_col:
            moved = self.game.move_piece(1)
            if not moved: break
            self._animate()

        while self.game.current_pos[1] > target_col:
            moved = self.game.move_piece(-1)
            if not moved: break
            self._animate()

        # keep soft-dropping until lock? not sure :(
        self.game.hard_drop()
        # while not self.game.game_over:
        #     moved_down = self.game.drop_piece()
        #     # self._maybe_animate()
        #     if not moved_down:
        #         break

        # reward calculation
        lines_cleared = self.game.lines_cleared - prev_lines
        height = get_board_height(self.game.board)
        holes = count_holes(self.game.board)
        bumpiness = get_bumpiness(self.game.board)

        reward = 0.1
        if lines_cleared > 0:
            line_rewards = [0.0, 1.0, 3.0, 5.0, 8.0]
            reward += line_rewards[min(lines_cleared, 4)]
        # if height_delta > 0:
        #     reward -= 0.05 * height_delta
        # if holes_delta > 0:
        #     reward -= 0.3 * holes_delta
        # if bumpiness_delta > 0:
        #     reward -= 0.05 * bumpiness_delta
        if self.game.game_over:
            reward -= 5.0
        
        state = self.game.render().astype(np.uint8)
        info = {"score": self.game.score, "lines_cleared": self.game.lines_cleared,
                "height": height, "holes": holes, "bumpiness": bumpiness, "lines_cleared_move": lines_cleared,
                "rotations": num_rotations, "column_used": target_col}
        return state, reward, self.game.game_over, False, info
    
    def close(self):
        self.game.close()
