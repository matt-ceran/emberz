import numpy as np

def build_weights():
    # Shapes
    W1 = np.zeros((32, 400), dtype=np.float32)
    b1 = np.zeros(32, dtype=np.float32)
    W2 = np.zeros((32, 32), dtype=np.float32)
    b2 = np.zeros(32, dtype=np.float32)
    W3 = np.zeros((6, 32), dtype=np.float32)
    b3 = np.zeros(6, dtype=np.float32)

    # First layer: ball_x (unit 0) and paddle_x (unit 1)
    scale = 0.1
    for y in range(10):
        for x in range(10):
            # ball channel = 1
            idx_ball = (y * 10 + x) * 4 + 1
            W1[0, idx_ball] = scale * x
            # paddle channel = 0, only at y=9
            if y == 9:
                idx_paddle = (y * 10 + x) * 4 + 0
                W1[1, idx_paddle] = scale * x

    # Second layer: unit 0 = tanh(ball_x_unit - paddle_x_unit)
    W2[0, 0] = 1.0
    W2[0, 1] = -1.0

    # Output layer
    K = 100.0
    # no-op (0)
    b3[0] = 0.1
    # left (1)
    W3[1, 0] = -K
    # up (2) - unused
    b3[2] = -100.0
    # right (3)
    W3[3, 0] = K
    # down (4) - unused
    b3[4] = -100.0
    # fire (5) - unused
    b3[5] = -100.0

    return {
        "W1": W1, "b1": b1,
        "W2": W2, "b2": b2,
        "W3": W3, "b3": b3
    }
