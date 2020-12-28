import contextlib
import random


@contextlib.contextmanager
def pyrandom_seed(seed, *addl_seeds):
    """Context manager which seeds the Python Random PRNG with the specified seed and
    restores the state afterward. Based on numpy_seed from fairseq.data.data_utils. """
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


