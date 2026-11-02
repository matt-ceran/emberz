import numpy as np


def build_weights():
    def idx(y, x, c):
        return (y * 10 + x) * 4 + c

    W1 = np.zeros((32, 400), dtype=np.float32)
    b1 = np.zeros(32, dtype=np.float32)
    W2 = np.zeros((32, 32), dtype=np.float32)
    b2 = np.zeros(32, dtype=np.float32)
    W3 = np.zeros((6, 32), dtype=np.float32)
    b3 = np.zeros(6, dtype=np.float32)

    # h1[0] ~= sign(ball_x - paddle_x); h1[1] ~= sign(paddle_x - ball_x)
    for y in range(10):
        for x in range(10):
            W1[0, idx(y, x, 1)] += 6.0 * x      # ball channel
            W1[0, idx(y, x, 0)] += -6.0 * x     # paddle channel
            W1[1, idx(y, x, 1)] += -6.0 * x
            W1[1, idx(y, x, 0)] += 6.0 * x

            # h1[2] ~= sign(ball_x - trail_x): horizontal velocity
            W1[2, idx(y, x, 1)] += 6.0 * x
            W1[2, idx(y, x, 2)] += -6.0 * x

            # h1[3] ~= sign(ball_y - trail_y): +1 descending, -1 ascending
            W1[3, idx(y, x, 1)] += 6.0 * y
            W1[3, idx(y, x, 2)] += -6.0 * y

    # Exact reset detectors: ball and trail coincide at (3,0) or (3,9).
    for row, x in ((4, 0), (5, 9)):
        W1[row, idx(3, x, 1)] = 8.0
        W1[row, idx(3, x, 2)] = 8.0
        b1[row] = -9.6

    # Pass through the comparison and reset features.
    W2[0, 0] = 1.0
    W2[1, 1] = 1.0
    W2[4, 4] = 1.0
    W2[5, 5] = 1.0

    # Descending-direction detectors from saturated velocity signs.
    # h2[2] active only for dx=+1, dy=+1; h2[3] only for dx=-1, dy=+1.
    W2[2, 2] = 8.0
    W2[2, 3] = 8.0
    b2[2] = -8.0
    W2[3, 2] = -8.0
    W2[3, 3] = 8.0
    b2[3] = -8.0

    # Strongly suppress actions that do nothing but are not the chosen no-op.
    b3[:] = 0.0
    b3[2] = -40.0
    b3[4] = -40.0
    b3[5] = -40.0
    b3[0] = 5.0      # no-op when aligned and not descending
    b3[1] = 0.0      # left
    b3[3] = 0.0      # right

    # Right: ball is to the right, or aligned while descending right, or reset from left edge.
    W3[3, 0] = 40.0
    W3[3, 2] = 18.0
    W3[3, 4] = 35.0

    # Left: ball is to the left, or aligned while descending left, or reset from right edge.
    W3[1, 1] = 40.0
    W3[1, 3] = 18.0
    W3[1, 5] = 35.0

    return {
        "W1": W1.astype(np.float32),
        "b1": b1.astype(np.float32),
        "W2": W2.astype(np.float32),
        "b2": b2.astype(np.float32),
        "W3": W3.astype(np.float32),
        "b3": b3.astype(np.float32),
    }
