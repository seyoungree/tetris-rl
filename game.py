	import numpy as np
	import pygame

	SHAPES = {
		'I': [[1, 1, 1, 1]],
		'O': [[1, 1], [1, 1]],
		'T': [[0, 1, 0], [1, 1, 1]],
		'S': [[0, 1, 1], [1, 1, 0]],
		'Z': [[1, 1, 0], [0, 1, 1]],
		'J': [[1, 0, 0], [1, 1, 1]],
		'L': [[0, 0, 1], [1, 1, 1]]
	}

	SHAPE_NAMES = list(SHAPES.keys())

	COLORS = {
		'I': (0, 240, 240),
		'O': (240, 240, 0),
		'T': (160, 0, 240),
		'S': (0, 240, 0),
		'Z': (240, 0, 0),
		'J': (0, 0, 240),
		'L': (240, 160, 0)
	}

	class TetrisGame:
		def __init__(self, width=10, height=20, block_size=20, 
					render_mode='rgb_array', gravity_fps=2, render_fps=30, queue_size=5):
			self.width, self.height = width, height
			self.block_size = block_size
			self.render_mode = render_mode

			# Gameplay timing
			self.gravity_fps = gravity_fps
			self.render_fps = render_fps
			self.gravity_interval = 1000 // self.gravity_fps

			# HUD / layout
			self.hud_height = 100
			self.sidebar_width = 6 * block_size
			self.screen_width = width * block_size + self.sidebar_width
			self.screen_height = height * block_size + self.hud_height

			self.queue_size = queue_size
			self.piece_queue = []                

			self.board = np.zeros((height, width), dtype=int)
			self.current_piece, self.current_shape = None, None
			self.current_shape_name = None
			self.current_pos = [0, 0]
			self.game_over = False
			self.score, self.lines_cleared = 0, 0
			self.lines_cleared_this_step = 0 

			pygame.init()
			if render_mode == 'human':
				self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
				pygame.display.set_caption("Tetris")
			elif render_mode == 'rgb_array':
				self.screen = pygame.Surface((self.screen_width, self.screen_height))
			
		def reset(self):
			self.board = np.zeros((self.height, self.width), dtype=int)
			self.game_over = False
			self.score = 0
			self.lines_cleared = 0
			self.lines_cleared_this_step = 0 
			self.piece_queue = [np.random.choice(SHAPE_NAMES) for _ in range(self.queue_size)]
			self.spawn_piece()
			if self.render_mode is not None:
				self.render()
		
		def spawn_piece(self):
			# Ensure queue exists
			if not hasattr(self, "piece_queue") or len(self.piece_queue) == 0:
				self.piece_queue = [np.random.choice(SHAPE_NAMES) for _ in range(self.queue_size)]

			# Take next piece from front of queue
			self.current_shape_name = self.piece_queue.pop(0)
			self.piece_queue.append(np.random.choice(SHAPE_NAMES))

			self.current_shape = np.array(SHAPES[self.current_shape_name])
			self.current_pos = [0, self.width // 2 - len(self.current_shape[0]) // 2]

			if not self.is_valid_position(self.current_shape, self.current_pos):
				self.game_over = True

		def is_valid_position(self, shape, pos):
			for y in range(len(shape)):
				for x in range(len(shape[0])):
					if shape[y][x]:
						board_y = pos[0] + y
						board_x = pos[1] + x
						
						# Check boundaries
						if board_x < 0 or board_x >= self.width or board_y >= self.height:
							return False
						
						# Check collision with existing pieces
						if board_y >= 0 and self.board[board_y][board_x]:
							return False
			return True
		
		def rotate_piece(self, clockwise=True):
			if clockwise:
				rotated = np.rot90(self.current_shape, k=-1)
			else:
				rotated = np.rot90(self.current_shape, k=1)
			
			if self.is_valid_position(rotated, self.current_pos):
				self.current_shape = rotated
				return True
			return False
		
		def move_piece(self, dx):
			new_pos = [self.current_pos[0], self.current_pos[1] + dx]
			if self.is_valid_position(self.current_shape, new_pos):
				self.current_pos = new_pos
				return True
			return False
		
		def drop_piece(self):
			new_pos = [self.current_pos[0] + 1, self.current_pos[1]]
			if self.is_valid_position(self.current_shape, new_pos):
				self.current_pos = new_pos
				return True
			else:
				self.lock_piece()
				return False
		
		def hard_drop(self):
			while self.drop_piece(): pass	
		
		def lock_piece(self):
			for y in range(len(self.current_shape)):
				for x in range(len(self.current_shape[0])):
					if self.current_shape[y][x]:
						board_y = self.current_pos[0] + y
						board_x = self.current_pos[1] + x
						if board_y >= 0:
							self.board[board_y][board_x] = SHAPE_NAMES.index(self.current_shape_name) + 1
			
			# Clear lines and spawn new piece
			lines = self.clear_lines()
			self.lines_cleared_this_step = lines
			self.spawn_piece()
			return lines
		
		def clear_lines(self):
			lines_to_clear = []
			for y in range(self.height):
				if np.all(self.board[y]):
					lines_to_clear.append(y)
			
			# Remove cleared lines from bottom to top to avoid index shifting bugs
			for y in sorted(lines_to_clear, reverse=True):
				self.board = np.delete(self.board, y, axis=0)
				self.board = np.vstack([np.zeros((1, self.width)), self.board])
			
			# Update score
			num_lines = len(lines_to_clear)
			if num_lines > 0:
				self.lines_cleared += num_lines
				self.score += [0, 100, 300, 500, 800][num_lines]
			
			return num_lines

		def render(self):
			self.screen.fill((0, 0, 0))

			font = pygame.font.SysFont("Arial", 20, bold=True)
			title_font = pygame.font.SysFont("Arial", 24, bold=True)
			title_surf = title_font.render("TETRIS", True, (255, 255, 255))
			self.screen.blit(title_surf, (10, 5))

			# scores & lines
			score_surf = font.render(f"Score: {self.score}", True, (200, 200, 200))
			lines_surf = font.render(f"Lines: {self.lines_cleared}", True, (200, 200, 200))
			self.screen.blit(score_surf, (10, 35))
			self.screen.blit(lines_surf, (10, 60))

			# border and board offset
			border_rect = pygame.Rect(0, self.hud_height, self.width * self.block_size, self.height * self.block_size)
			pygame.draw.rect(self.screen, (255, 255, 255), border_rect, 2)

			# grid
			grid_color = (40, 40, 40)
			for y in range(self.height):
				for x in range(self.width):
					rect = (x * self.block_size, y * self.block_size + self.hud_height, self.block_size, self.block_size)
					pygame.draw.rect(self.screen, grid_color, rect, 1)

			# locked pieces
			for y in range(self.height):
				for x in range(self.width):
					if self.board[y][x]:
						cell = int(self.board[y][x])
						shape_idx = cell - 1; shape_name = SHAPE_NAMES[shape_idx]
						color = COLORS[shape_name]
						rect = (x * self.block_size, y * self.block_size + self.hud_height,
								self.block_size, self.block_size)
						pygame.draw.rect(self.screen, color, rect)
						pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)

			# current piece
			if not self.game_over:
				color = COLORS[self.current_shape_name]
				for y in range(len(self.current_shape)):
					for x in range(len(self.current_shape[0])):
						if self.current_shape[y][x]:
							sx = (self.current_pos[1] + x) * self.block_size
							sy = (self.current_pos[0] + y) * self.block_size + self.hud_height
							rect = (sx, sy, self.block_size, self.block_size)
							pygame.draw.rect(self.screen, color, rect)
							pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)

			# next pieces queue
			sidebar_x = self.width * self.block_size + 10
			next_label = font.render("Next:", True, (200, 200, 200))
			self.screen.blit(next_label, (sidebar_x, self.hud_height - 30))

			preview_block = max(8, self.block_size // 2)
			for idx, shape_name in enumerate(self.piece_queue[:self.queue_size]):
				shape = SHAPES[shape_name]
				color = COLORS[shape_name]

				box_top = self.hud_height + idx * (preview_block * 3 + 10)
				box_height = preview_block * 3
				box_width = preview_block * 4
				box_rect = pygame.Rect(sidebar_x, box_top, box_width, box_height)
				pygame.draw.rect(self.screen, (80, 80, 80), box_rect, 1)

				# Compute offset so the shape is roughly centered in the box
				shape_h = len(shape)
				shape_w = len(shape[0])
				offset_x = sidebar_x + (box_width - shape_w * preview_block) // 2
				offset_y = box_top + (box_height - shape_h * preview_block) // 2

				for y in range(shape_h):
					for x in range(shape_w):
						if shape[y][x]:
							rect = (offset_x + x * preview_block, offset_y + y * preview_block,
									preview_block, preview_block)
							pygame.draw.rect(self.screen, color, rect)
							pygame.draw.rect(self.screen, (30, 30, 30), rect, 1)

			if self.render_mode == 'human':
				pygame.display.flip()
			return self._get_rgb_array()
		
		def _get_rgb_array(self):
			rgb_array = pygame.surfarray.array3d(self.screen)
			rgb_array = np.transpose(rgb_array, (1, 0, 2))
			return rgb_array
		
		def get_state_vector(self):
			# Calculate column heights (boundaries)
			boundaries = np.zeros(self.width, dtype=np.float32)
			for col in range(self.width):
				col_cells = self.board[:, col]
				if col_cells.any():
					top_filled = np.argmax(col_cells > 0)
					boundaries[col] = self.height - top_filled
			
			# Calculate holes per column
			holes = np.zeros(self.width, dtype=np.float32)
			for col in range(self.width):
				col_cells = self.board[:, col]
				if col_cells.any():
					top_filled = np.argmax(col_cells > 0)
					for row in range(top_filled + 1, self.height):
						if col_cells[row] == 0:
							holes[col] += 1
			
			if self.width > 1:
				relative_heights = np.abs(np.diff(boundaries))  # w-1 values
			else:
				relative_heights = np.array([0.0])
			
			# Falling piece one-hot encoding
			piece_encoding = np.zeros(len(SHAPE_NAMES), dtype=np.float32)
			piece_idx = SHAPE_NAMES.index(self.current_shape_name)
			piece_encoding[piece_idx] = 1.0
			next_piece_encoding = np.zeros(len(SHAPE_NAMES), dtype=np.float32)
			next_name = self.piece_queue[0]
			next_idx = SHAPE_NAMES.index(next_name)
			next_piece_encoding[next_idx] = 1.0
			
			# Lines cleared this step (normalized)
			cleared_lines = np.array([self.lines_cleared_this_step / 4.0], dtype=np.float32)
			
			# Additional useful features
			max_height = np.max(boundaries) if boundaries.any() else 0.0
			total_holes = np.sum(holes)
			avg_height = np.mean(boundaries)
			
			# Normalize features
			boundaries_norm = boundaries / self.height  # Normalize to [0, 1]
			holes_norm = holes / self.height
			relative_heights_norm = relative_heights / self.height
			
			additional_features = np.array([
				max_height / self.height,  # Normalized max height
				total_holes / (self.width * self.height),  # Normalized total holes
				avg_height / self.height,  # Normalized average height
			], dtype=np.float32)
			
			state = np.concatenate([
				holes_norm,
				boundaries_norm,
				# relative_heights_norm,
				piece_encoding,
				cleared_lines,
				# additional_features
			])
			
			return state

		def get_state_matrix(self):
			h, w = self.height, self.width

			# Channel 0: locked blocks
			locked = (self.board > 0).astype(np.float32)

			# Channel 1: current falling piece
			current = np.zeros_like(locked, dtype=np.float32)
			for y in range(len(self.current_shape)):
				for x in range(len(self.current_shape[0])):
					if self.current_shape[y][x]:
						by = self.current_pos[0] + y
						bx = self.current_pos[1] + x
						if 0 <= by < h and 0 <= bx < w:
							current[by][bx] = 1.0
			
			# Channel 2: boundaries
			boundaries = np.zeros_like(locked, dtype=np.float32)
			for col in range(w):
				col_cells = locked[:, col]
				if col_cells.any():
					top_filled = np.argmax(col_cells)
					height_val = h - top_filled  # Height from bottom
				else:
					height_val = 0
				boundaries[:, col] = height_val / h

			# Channel 3: Holes per column
			holes_channel = np.zeros_like(locked, dtype=np.float32)
			for col in range(w):
				col_cells = locked[:, col]
				if col_cells.any():
					top_filled = np.argmax(col_cells)
					num_holes = 0
					for row in range(top_filled + 1, h):
						if col_cells[row] == 0:
							num_holes += 1
					holes_channel[:, col] = num_holes / h

			state = np.stack([locked, current, boundaries, holes_channel], axis=0)
			return state
		
		def close(self):
			pygame.quit()

		def play_manual(self):
			clock = pygame.time.Clock()
			running = True
			last_gravity_time = pygame.time.get_ticks()

			while running:
				if self.game_over: break

				for event in pygame.event.get():
					if event.type == pygame.QUIT:
						running = False
					if event.type == pygame.KEYDOWN:
						if event.key in (pygame.K_ESCAPE, pygame.K_q):
							running = False
						elif event.key == pygame.K_LEFT:
							self.move_piece(-1)
						elif event.key == pygame.K_RIGHT:
							self.move_piece(1)
						elif event.key == pygame.K_UP:
							self.rotate_piece()
						elif event.key == pygame.K_DOWN:
							self.drop_piece()
						elif event.key == pygame.K_SPACE:
							self.hard_drop()

				now = pygame.time.get_ticks()
				if not self.game_over and now - last_gravity_time >= self.gravity_interval:
					self.drop_piece()
					last_gravity_time = now

				self.render()
				clock.tick(self.render_fps)
			self.close()

	if __name__ == "__main__":
		game = TetrisGame(render_mode='rgb_array')
		game.reset()
		game.play_manual()
