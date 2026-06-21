import numpy as np
def win_distance(frame):
    # placeholder mirror of the mined clear_color(6) goal
    return float((np.asarray(frame) == 6).sum())
