import numpy as np

from sortedcollections import SortedDict
from scipy.linalg import pinv2, cholesky, inv
from scipy import outer, dot, multiply, zeros, diag, mat, sum


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))]
    which returns ranks in [1, len(x)].
    (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """
    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return -weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


class GA:
    """
    Basic population based genetic algorithm
    """

    def __init__(self, num_params,
                 pop_size=100,
                 elite_frac=0.1,
                 mut_rate=0.9,
                 mut_amp=0.1):

        # misc
        self.num_params = num_params
        self.pop_size = pop_size
        self.n_elites = int(self.pop_size * elite_frac)

        # individuals
        self.individuals = np.random.normal((pop_size, num_params))
        self.fitness = SortedDict({})

        # mutations
        self.mut_amp = mut_amp
        self.mut_rate = mut_rate
        self.mut_ind = [i for i in range(self.pop_size)]

    def ask(self, pop_size):
        """
        Returns the newly created individual(s)
        """
        return np.copy(self.individuals[self.mut_ind])

    def tell(self, individual, score):
        """
        Updates the population
        """
        assert(len(scores) == len(individual)
               ), "Inconsistent reward_table size reported."

        # add new fitness evaluations
        for ind, s in zip(self.mut_ind, score):
            self.fitness[s, ind] = s
        self.fitness.update()

        # tournament selection
        self.mut_ind = []
        sorted_keys = self.fitness.keys()[:self.pop_size - self.n_elites]
        for i in range(self.pop_size - self.n_elites):
            k, l = np.random.choice(sorted_keys, size=2, replace=False)
            if self.fitness[k] > self.fitness[l]:
                self.mut_ind.append(k[0])
            else:
                self.mut_ind.append(l[0])

        # mutation
        for ind in self.mut_ind:
            u = np.random.rand()
            if u < self.mut_rate:
                params = self.individuals[ind]
                noise = np.random.normal(loc=0, scale=self.mut_amp * params)
                params += noise
