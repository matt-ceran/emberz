import numpy as np

def build_weights():
    def idx(y, x, c):
        return (y * 10 + x) * 4 + c

    W1 = np.zeros((32, 400), dtype=np.float32)
    b1 = np.zeros((32,), dtype=np.float32)
    W2 = np.zeros((32, 32), dtype=np.float32)
    b2 = np.zeros((32,), dtype=np.float32)
    W3 = np.zeros((6, 32), dtype=np.float32)
    b3 = np.zeros((6,), dtype=np.float32)

    alpha = 4.0
    # h1[0] = tanh(alpha * (ball_column - paddle_column))
    # ball channel = 1: weight +alpha*x for that column (any row; only one ball cell is on)
    for y in range(10):
        for x in range(10):
            W1[0, idx(y, x, 1)] = alpha * x
    # paddle channel = 0 on row y=9: weight -alpha*x
    for x in range(10):
        W1[0, idx(9, x, 0)] = -alpha * x

    # h2[0] = tanh(3 * h1[0]): re-saturate the sign, preserve 0 at tie
    W2[0, 0] = 3.0

    # logits: right (3) = +15*sign, left (1) = -15*sign
    W3[3, 0] = 15.0
    W3[1, 0] = -15.0
    # no-op actions {0,2,4,5} biased up so a tie (Delta=0) deterministically stays put
    for a in (0, 2, 4, 5):
        b3[a] = 6.0

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
