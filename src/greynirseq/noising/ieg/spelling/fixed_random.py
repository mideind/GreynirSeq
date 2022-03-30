import random as python_random


RANDOM_STATE = 0

def random():
    """Simple LCG random number generator for reproducibility"""
    global RANDOM_STATE

    # Parameters picked at random from wikipedia, probably not robust
    m = 2**32
    a = 1664525
    c = 1013904223

    RANDOM_STATE = (a * RANDOM_STATE + c) % m

    return RANDOM_STATE


def set_random_state(new_state):
    global RANDOM_STATE
    RANDOM_STATE = new_state


def exp_len(exponent, min_len=1, max_len=5):
    res = min_len
    while coinflip(exponent) and res < max_len:
        res += 1
    return res


def coinflip(chance):
    return python_random.random() < chance
