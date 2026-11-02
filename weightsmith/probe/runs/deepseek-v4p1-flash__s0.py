import numpy as np

def build_weights():
    W1 = np.zeros((32, 400), dtype=np.float32)
    b1 = np.zeros(32, dtype=np.float32)
    W2 = np.zeros((32, 32), dtype=np.float32)
    b2 = np.zeros(32, dtype=np.float32)
    W3 = np.zeros((6, 32), dtype=np.float32)
    b3 = np.zeros(6, dtype=np.float32)

    def idx(y, x, c):
        return (y * 10 + x) * 4 + c

    # h1[0]: ball_x
    for y in range(10):
        for x in range(10):
            W1[0, idx(y, x, 1)] = 0.01 * x
    # h1[1]: trail_x
    for y in range(10):
        for x in range(10):
            W1[1, idx(y, x, 2)] = 0.01 * x
    # h1[2]: paddle_x
    for x in range(10):
        W1[2, idx(9, x, 0)] = 0.01 * x
    # h1[3]: left wall bounce indicator
    for y in range(10):
        W1[3, idx(y, 0, 1)] = 10.0
        W1[3, idx(y, 1, 2)] = 10.0
    b1[3] = -15.0
    # h1[4]: right wall bounce indicator
    for y in range(10):
        W1[4, idx(y, 9, 1)] = 10.0
        W1[4, idx(y, 8, 2)] = 10.0
    b1[4] = -15.0

    # h2[0]: target_x (scaled by 0.05)
    W2[0, 0] = 10.0
    W2[0, 1] = -5.0
    W2[0, 3] = 0.05
    W2[0, 4] = -0.05
    b2[0] = 0.0

    # h2[1]: paddle_x (scaled by 0.05)
    W2[1, 2] = 5.0
    b2[1] = 0.0

    # Output logits
    # noop
    b3[0] = 5.0
    # left
    W3[1, 0] = -1000.0
    W3[1, 1] = 1000.0
    # right
    W3[3, 0] = 1000.0
    W3[3, 1] = -1000.0
    # other actions (u, d, f)
    b3[2] = -100.0
    b3[4] = -100.0
    b3[5] = -100.0

    return {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3
    }
