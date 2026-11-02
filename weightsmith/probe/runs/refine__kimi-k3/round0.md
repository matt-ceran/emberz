Design: the first layer turns the one-hot ball/paddle/trail channels into three saturated signed features: “ball is right of paddle”, “ball is left of paddle”, and “ball is falling” (`ball_y > trail_y`). The second layer forms AND/NAND logic: move-right only if falling and ball is right, move-left only if falling and ball is left, otherwise no-op. This tracks the ball column while it descends; on the last row it places the paddle on the ball’s current cell, which the environment accepts as a hit. Output weights are large so softmax sampling is effectively deterministic, with `u/d/f` suppressed.

```python
import numpy as np

def build_weights():
    B = 8.0    # input feature scale / comparator sharpness
    K = 30.0   # hidden AND/NAND sharpness
    A = 35.0   # output logit scale

    W1 = np.zeros((32, 400), dtype=np.float32)
    b1 = np.zeros((32,), dtype=np.float32)
    W2 = np.zeros((32, 32), dtype=np.float32)
    b2 = np.zeros((32,), dtype=np.float32)
    W3 = np.zeros((6, 32), dtype=np.float32)
    b3 = np.zeros((6,), dtype=np.float32)

    def idx(y, x, c):
        return (y * 10 + x) * 4 + c

    # h1[0]: +1 iff ball_x > paddle_x
    # h1[1]: +1 iff paddle_x > ball_x
    # h1[2]: +1 iff ball_y > trail_y  (falling)
    for y in range(10):
        for x in range(10):
            W1[0, idx(y, x, 1)] += B * x
            W1[1, idx(y, x, 1)] += -B * x
            W1[2, idx(y, x, 1)] += B * y
            W1[2, idx(y, x, 2)] += -B * y
    for x in range(10):
        W1[0, idx(9, x, 0)] += -B * x
        W1[1, idx(9, x, 0)] += B * x
    b1[0] = -0.5 * B
    b1[1] = -0.5 * B
    b1[2] = -0.5 * B

    # h2[0]: right_gate = h1[0] AND h1[2]
    # h2[1]: left_gate  = h1[1] AND h1[2]
    # h2[2]: noop_gate  = NOT( h1[2] AND (h1[0] OR h1[1]) )
    W2[0, 0] = K
    W2[0, 2] = K
    b2[0] = -K

    W2[1, 1] = K
    W2[1, 2] = K
    b2[1] = -K

    W2[2, 0] = -K
    W2[2, 1] = -K
    W2[2, 2] = -K

    # Actions: 0='n', 1='l', 3='r'. Suppress 2,4,5.
    W3[0, 2] = A
    W3[1, 1] = A
    W3[3, 0] = A
    b3[2] = -A
    b3[4] = -A
    b3[5] = -A

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
```