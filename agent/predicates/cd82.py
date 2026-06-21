import numpy as np

# cd82 win condition (mined from cd82.py :: wvrremwltt, lines 718-731):
#   canvas sprite "xytrjjbyib" (10x10) must match target sprite "eoqnvkspoa-*" (10x10)
#   on every cell EXCEPT the two diagonals. next_level() fires when:
#       np.array_equal(canvas[mask], target[mask])
#   where mask[i,j] is False on the main diagonal (i==j) and anti-diagonal (j==9-i).
#
# Frame layout (verified on real 64x64 frames):
#   target "eoqnvkspoa-*" lives at level position (3, 3)   -> frame rows 3:13, cols 3:13
#   canvas "xytrjjbyib"   lives at level position (27, 34) -> frame rows 34:44, cols 27:37
#
# win_distance = count of OFF-DIAGONAL cells where canvas != target.
#   0.0 exactly at the goal; +1 per still-wrong cell -> smooth & monotone.

# off-diagonal mask, identical to poqpfcjieu in the source
_MASK = np.ones((10, 10), dtype=bool)
for _i in range(10):
    _MASK[_i, _i] = False
    _MASK[_i, 9 - _i] = False

# target slice
_TR0, _TR1, _TC0, _TC1 = 3, 13, 3, 13
# canvas slice
_CR0, _CR1, _CC0, _CC1 = 34, 44, 27, 37


def win_distance(frame) -> float:
    f = np.asarray(frame)
    if f.shape[0] < _CR1 or f.shape[1] < _CC1:
        # frame too small to contain both regions -> treat as maximally far
        return float(_MASK.sum())
    target = f[_TR0:_TR1, _TC0:_TC1]
    canvas = f[_CR0:_CR1, _CC0:_CC1]
    if target.shape != (10, 10) or canvas.shape != (10, 10):
        return float(_MASK.sum())
    mismatch = (canvas != target) & _MASK
    return float(np.count_nonzero(mismatch))
