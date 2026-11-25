"""ts_advise_sub: the advisor's card choices graded from the position before the action (replayed choices, forced
option). Workshop and Remodel positions on turn 3; an impossible replay is refused."""
import ctypes
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ts import lib as T  # noqa: E402

core = T.core
L = T.lib
_I, _F, _U8 = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_ubyte)
L.ts_advise_sub.restype = ctypes.c_int
L.ts_advise_sub.argtypes = [_F, ctypes.c_int, ctypes.c_int, _I, ctypes.c_int, _I, _I, _I, ctypes.c_int, ctypes.c_int,
                            ctypes.c_int, ctypes.c_int, ctypes.c_int, _I, _I, ctypes.c_int, ctypes.c_int, _I,
                            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_uint64, _F, _F, _U8, _F]
ID, NA = core.ID, core.NA
K = ["Village", "Smithy", "Market", "Militia", "Workshop", "Remodel", "Cellar", "Chapel", "Laboratory", "Witch"]
C2 = core.params("rl3_c2_2654.npy")
NH, NF = core.net_shape(C2)
DE_REMODEL_TRASH, DE_REMODEL_GAIN, DE_WORKSHOP_GAIN = 7, 8, 9


def ints(xs):
    return (ctypes.c_int * max(len(xs), 1))(*xs)


def supply():
    s = [-1] * core.NUM_CARDS
    for c, n in {"Copper": 46, "Silver": 40, "Gold": 30, "Estate": 8, "Duchy": 8, "Province": 8, "Curse": 10}.items():
        s[ID[c]] = n
    for c in K:
        s[ID[c]] = 10
    return s


def ids(*n):
    return [ID[x] for x in n]


def ask(hand, card, hist, kind, opts, may_pass, inc, secs=20.0):
    T.configure(k0=24, k1=24, act=1, obs=1)
    deck = ids(*["Copper"] * 5, "Estate", "Silver")
    zones = [deck, hand, [], [], ids(*["Copper"] * 7, "Estate", "Estate", "Estate", "Silver", "Militia"), [], [], []]
    flat = [c for z in zones for c in z]
    q, se = np.full(NA, np.nan, np.float32), np.full(NA, np.nan, np.float32)
    mask, out = np.zeros(NA, np.uint8), np.zeros(5, np.float32)
    k = L.ts_advise_sub(C2.ctypes.data_as(_F), NH, NF, ints(supply()), 0, ints(flat), ints([len(z) for z in zones]),
                        ints([2, 2]), 0, 1, 1, 0, ID[card], ints([h[0] for h in hist]), ints([h[1] for h in hist]),
                        len(hist), kind, ints(opts), len(opts), may_pass, inc, secs, 7, q.ctypes.data_as(_F),
                        se.ctypes.data_as(_F), mask.ctypes.data_as(_U8), out.ctypes.data_as(_F))
    return k, q, se, mask.astype(bool)


def test_workshop_gain():
    opts = [i for i, c in enumerate(supply()) if c > 0 and core.lib.rl_num_actions() and _cost(i) <= 4]
    k, q, se, m = ask(ids("Workshop", "Copper", "Copper", "Estate", "Copper"), "Workshop", [], DE_WORKSHOP_GAIN,
                      opts, 0, ID["Silver"])
    assert k == 24, k
    assert set(np.where(m)[0]) == set(opts)
    assert all(0 <= q[a] <= 1 for a in opts)
    assert q[ID["Curse"]] < q[ID["Silver"]]                      # gaining a Curse is worse than a Silver


def test_remodel_gain_after_replayed_trash():
    opts = [i for i, c in enumerate(supply()) if c > 0 and _cost(i) <= 4]
    k, q, se, m = ask(ids("Remodel", "Copper", "Copper", "Estate", "Copper"), "Remodel",
                      [(DE_REMODEL_TRASH, ID["Estate"])], DE_REMODEL_GAIN, opts, 0, ID["Silver"])
    assert k == 24, k


def test_replay_that_cannot_happen_is_refused():
    opts = [i for i, c in enumerate(supply()) if c > 0 and _cost(i) <= 5]
    k, *_ = ask(ids("Remodel", "Copper", "Copper", "Copper", "Copper"), "Remodel",       # no Estate in hand to trash
                [(DE_REMODEL_TRASH, ID["Estate"])], DE_REMODEL_GAIN, opts, 0, ID["Silver"])
    assert k == -1, k


def _cost(i):
    from dominion_engine.cards import CARDS
    return CARDS[i].cost if hasattr(CARDS[i], "cost") else CARDS[i]["cost"]


if __name__ == "__main__":
    for f in (test_workshop_gain, test_remodel_gain_after_replayed_trash, test_replay_that_cannot_happen_is_refused):
        f()
        print(f.__name__, "ok")
