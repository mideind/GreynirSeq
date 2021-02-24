import numpy as np


class LinearSampler:
    def __init__(self, start_map, end_map, start_step, end_step, seed=1):
        """
        when step < step_start
            start_ps  (0)
            start_go  (1)

        when step_start < step < step_end
            t = (curr - start)/(end - start)
            ps = t * (end_ps - start_ps) + start_ps
            go = t * (end_go - start_go) + start_go

        after step_end
            ps = end_ps (1)
            go = end_go (0)

        # (not is_linear) * step_start + is_linear * scaling
        # is_linear = (step >= step_start)


        Example usage for pseudo-labeling:
            self.sampling_fn = LinearSampler(
                {"pseudo": 1.0, "gold": 0.0},
                {"pseudo": 1.0, "gold": 9.0},
                2000,
                7000,
            )

        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.start_map = start_map
        self.end_map = end_map
        self.start_step = start_step
        self.end_step = end_step
        self.num_updates = 0

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates

    def __call__(self, keys):
        current_weights = {}
        t = (self.num_updates - self.start_step) / ((self.end_step - self.start_step) or 1)
        for key in self.start_map:
            start_val, end_val = self.start_map[key], self.end_map[key]
            if self.num_updates <= self.start_step:
                current_weights[key] = start_val
            elif self.num_updates <= self.end_step:
                current_weights[key] = t * (end_val - start_val) + start_val
            else:
                current_weights[key] = end_val
        distr = [current_weights[key] for key in keys]
        total = sum(distr)
        distr = [d / total for d in distr]
        return np.random.choice(keys, p=distr)


def make_sampling_fn(weight_map, seed=1):
    # map = {
    #    "pseudo": 1.0,
    #    "gold": 10.0,
    # }
    rng = np.random.RandomState(seed)
    total = sum(weight_map.values())

    def sampling_fn(keys):
        distribution = [weight_map[key] / total for key in keys]
        return rng.choice(keys, p=distribution)

    return sampling_fn
