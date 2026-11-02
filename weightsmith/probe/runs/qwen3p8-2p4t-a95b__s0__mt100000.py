import numpy as np


def build_weights():
    W1 = np.zeros((32, 400), dtype=np.float32)
    b1 = np.zeros(32, dtype=np.float32)
    W2 = np.zeros((32, 32), dtype=np.float32)
    b2 = np.zeros(32, dtype=np.float32)
    W3 = np.zeros((6, 32), dtype=np.float32)
    b3 = np.zeros(6, dtype=np.float32)

    # Large gain for near-step activations and deterministic logits.
    K = 20.0
    A = 20.0

    def idx(y, x, c):
        return (y * 10 + x) * 4 + c

    # -------------------------------------------------------------------------
    # Hidden layer 1
    #
    # Units 0..8:
    #   Row detectors: +1 if ball_y == r, else -1.
    #
    # Units 9..17:
    #   Right-move detectors for each row:
    #   active if projected_x - paddle_x > 0.
    #
    # Units 18..26:
    #   Left-move detectors for each row:
    #   active if projected_x - paddle_x < 0.
    # -------------------------------------------------------------------------
    for r in range(9):
        u = r
        for y in range(10):
            val = K if y == r else -K
            for x in range(10):
                W1[u, idx(y, x, 1)] = val  # ball channel
        b1[u] = 0.0

    for r in range(9):
        # Lookahead distance. Exact remaining vertical distance is 9-r,
        # but cap at 4 for stability near walls.
        L = min(4, 9 - r)

        u_right = 9 + r
        u_left = 18 + r

        for y in range(10):
            for x in range(10):
                # projected_x = ball_x + L * (ball_x - trail_x)
                # plus wall correction:
                #   if ball_x == 0, assume it will leave wall to the right: +L
                #   if ball_x == 9, assume it will leave wall to the left:  -L
                ball_coeff = (1.0 + L) * x
                if x == 0:
                    ball_coeff += L
                elif x == 9:
                    ball_coeff -= L

                trail_coeff = -L * x
                paddle_coeff = -x

                # Right condition: projected_x - paddle_x > 0.
                W1[u_right, idx(y, x, 1)] = K * ball_coeff     # ball
                W1[u_right, idx(y, x, 2)] = K * trail_coeff    # trail
                W1[u_right, idx(y, x, 0)] = K * paddle_coeff   # paddle
                b1[u_right] = -0.5 * K

                # Left condition: projected_x - paddle_x < 0.
                W1[u_left, idx(y, x, 1)] = -K * ball_coeff
                W1[u_left, idx(y, x, 2)] = -K * trail_coeff
                W1[u_left, idx(y, x, 0)] = -K * paddle_coeff
                b1[u_left] = -0.5 * K

    # -------------------------------------------------------------------------
    # Hidden layer 2
    #
    # Units 0..8:
    #   AND(row_r, right_condition_r)
    #
    # Units 9..17:
    #   AND(row_r, left_condition_r)
    #
    # With tanh-coded booleans (+1 true, -1 false), AND(a,b) is implemented as
    # tanh(K * (a + b - 1)).
    # -------------------------------------------------------------------------
    K_and = 20.0

    for r in range(9):
        right_and = r
        left_and = 9 + r

        W2[right_and, r] = K_and          # row detector
        W2[right_and, 9 + r] = K_and      # right detector for this row
        b2[right_and] = -K_and

        W2[left_and, r] = K_and           # row detector
        W2[left_and, 18 + r] = K_and      # left detector for this row
        b2[left_and] = -K_and

    # -------------------------------------------------------------------------
    # Output logits
    #
    # Action 1 = 'l', action 3 = 'r'.
    #
    # There are 9 row-specific AND units per direction. If all are false, their
    # outputs are about -1, so the bias 9*A cancels them to about 0. If exactly
    # one is true, the logit becomes about 2*A.
    #
    # No-op is favored when neither left nor right is active.
    # -------------------------------------------------------------------------
    for r in range(9):
        W3[3, r] = A          # right action from right AND units
        W3[1, 9 + r] = A      # left action from left AND units

    b3[3] = 9.0 * A
    b3[1] = 9.0 * A

    b3[0] = 10.0   # no-op
    b3[2] = -10.0  # up, unused
    b3[4] = -10.0  # down, unused
    b3[5] = -10.0  # fire, unused

    return {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3,
    }
