import numpy as np

def count_holes(board):
    holes = 0
    width = board.shape[1]

    for x in range(width):
        block_found = False
        for y in range(board.shape[0]):
            if board[y][x]:
                block_found = True
            elif block_found and board[y][x] == 0:
                holes += 1
    return holes

def row_transitions(board):
    """Count transitions full->empty or empty->full across each row."""
    transitions = 0
    for y in range(board.shape[0]):
        prev = 1   # treat outside as filled
        for x in range(board.shape[1]):
            curr = 1 if board[y][x] else 0
            if curr != prev:
                transitions += 1
            prev = curr
        # right boundary
        if prev == 0:
            transitions += 1
    return transitions


def column_transitions(board):
    """Count transitions in each column top->bottom."""
    transitions = 0
    for x in range(board.shape[1]):
        prev = 1
        for y in range(board.shape[0]):
            curr = 1 if board[y][x] else 0
            if curr != prev:
                transitions += 1
            prev = curr
        if prev == 0:
            transitions += 1
    return transitions


def wells(board):
    """Cumulative well depth."""
    h, w = board.shape
    well_sum = 0

    for x in range(w):
        for y in range(h):
            if board[y][x] != 0:
                continue

            left_filled = (x == 0 or board[y][x-1] != 0)
            right_filled = (x == w-1 or board[y][x+1] != 0)

            if left_filled and right_filled:
                depth = 1
                yy = y + 1
                while yy < h and board[yy][x] == 0:
                    depth += 1
                    yy += 1
                well_sum += depth

    return well_sum


def landing_height(board, piece_final_y, piece_height, board_height):
    """Height from bottom at which the piece was placed."""
    return board_height - (piece_final_y + piece_height/2.0)


def eroded_cells(cleared_lines, piece_blocks_filled_those_rows):
    return cleared_lines * piece_blocks_filled_those_rows