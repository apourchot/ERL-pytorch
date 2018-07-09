# adapted from https://github.com/hardmaru/estool/blob/master/es.py
import numpy as np

from Optimizers import Adam, SGD, BasicSGD
from pybrain.utilities import flat2triu, triu2flat
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


class OpenES:

    """
    Basic Version of OpenAI Evolution Strategies
    """

    def __init__(self, num_params,
                 optimizer_class,
                 mu_init=None,
                 sigma_init=0.1,
                 lr=10**-2,
                 pop_size=256,
                 antithetic=False,
                 weight_decay=0.005,
                 rank_fitness=True):

        # misc
        self.num_params = num_params
        self.first_interation = True

        # distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)
        self.sigma = sigma_init

        # optimizarion stuff
        self.learning_rate = lr
        self.optimizer = optimizer_class(self.learning_rate)

        # sampling stuff
        self.pop_size = pop_size
        self.antithetic = antithetic
        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness

    def ask(self, pop_size):
        """
        Returns a list of candidates parameterss
        """
        if self.antithetic and not pop_size % 2:
            epsilon_half = np.random.randn(self.pop_size // 2, self.num_params)
            epsilon = np.concatenate([epsilon_half, - epsilon_half])

        else:
            epsilon = np.random.randn(pop_size, self.num_params)

        return self.mu + epsilon * self.sigma

    def tell(self, solutions, scores):
        """
        Updates the distribution
        """
        assert(len(scores) ==
               self.pop_size), "Inconsistent reward_table size reported."

        reward = np.array(scores)
        if self.rank_fitness:
            reward = compute_centered_ranks(reward)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, solutions)
            reward += l2_decay

        epsilon = (solutions - self.mu) / self.sigma
        grad = -1/(self.sigma * self.pop_size) * np.dot(reward, epsilon)

        # optimization step
        self.optimizer.stepsize = self.learning_rate
        step = self.optimizer.step(grad)
        self.mu += step

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and sigma
        """
        return np.copy(self.mu), np.copy(self.sigma ** 2)


class SNES():
    """
    Separable NES (diagonal), as described in Schaul,
    Glasmachers and Schmidhuber (GECCO'11)
    """

    def __init__(self, num_params,
                 optimizer_class,
                 mu_init=None,
                 sigma_init=0.01,
                 lr=10**-2,
                 pop_size=256,
                 antithetic=True,
                 weight_decay=0.01,
                 rank_fitness=True):

        # misc
        self.num_params = num_params

        # distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)
        self.sigma = sigma_init * np.ones(self.num_params)

        # optimizarion stuff
        self.learning_rate = lr
        self.optimizer = optimizer_class(lr)

        # sampling stuff
        self.pop_size = pop_size
        self.antithetic = antithetic
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness

    def ask(self, pop_size):
        if self.antithetic and not pop_size % 2:
            epsilon_half = np.random.randn(self.pop_size // 2, self.num_params)
            epsilon = np.concatenate([epsilon_half, - epsilon_half])

        else:
            epsilon = np.random.randn(pop_size, self.num_params)
        return self.mu + self.sigma * epsilon

    def tell(self, solutions, scores):

        assert(len(scores) ==
               self.pop_size), "Inconsistent reward_table size reported."

        # scores preprocess
        reward = np.array(scores)
        if self.rank_fitness:
            reward = compute_centered_ranks(reward)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, solutions)
            reward += l2_decay

        # computing gradient
        epsilon = (solutions - self.mu) / self.sigma
        grad = np.zeros(2 * self.num_params)
        grad[:self.num_params] = -1/self.pop_size * \
            self.sigma * np.dot(reward, epsilon)
        grad[self.num_params:] = -1/self.pop_size * 0.5 * \
            np.dot(reward, [s ** 2 - 1 for s in epsilon])

        # optimization step
        step = self.optimizer.step(grad)
        self.mu += step[:self.num_params]
        self.sigma = self.sigma * np.exp(step[self.num_params:])

    def get_distrib_params(self):
        return np.copy(self.mu), np.copy(self.sigma ** 2)


class CEM:

    """
    Cross-entropy methods.
    """

    def __init__(self, num_params,
                 mu_init=None,
                 sigma_init=0.1,
                 pop_size=256,
                 antithetic=False,
                 weight_decay=0.01,
                 rank_fitness=True):

        # misc
        self.num_params = num_params

        # distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)
        self.sigma = sigma_init
        self.cov = self.sigma ** 2 * np.eye(self.num_params)
        self.coord = np.eye(self.num_params)
        self.diag = self.sigma ** 2 * np.ones(self.num_params)

        # sampling stuff
        self.pop_size = pop_size
        self.antithetic = antithetic
        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness
        self.parents = pop_size // 2
        self.weights = np.ones(self.parents)
        self.weights /= self.weights.sum()

    def ask(self, pop_size):
        """
        Returns a list of candidates parameterss
        """
        if self.antithetic:
            epsilon_half = np.random.randn(self.pop_size // 2, self.num_params)
            epsilon = np.concatenate([epsilon_half, - epsilon_half])

        else:
            epsilon = np.random.randn(self.num_params, pop_size)

        return (self.coord @ np.diag(np.sqrt(self.diag)) @ epsilon).T + self.mu

    def tell(self, solutions, scores):
        """
        Updates the distribution
        """
        assert(len(scores) ==
               self.pop_size), "Inconsistent reward_table size reported."

        # scores preprocess
        reward = np.array(scores)
        if self.rank_fitness:
            reward = compute_centered_ranks(reward)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, solutions)
            reward += l2_decay

        idx_sorted = np.argsort(reward)

        old_mu = np.copy(self.mu)
        self.mu = self.weights @ solutions[idx_sorted[-self.parents:]]

        X = solutions[idx_sorted[-self.parents:]] - old_mu
        self.cov = 1/self.parents * X.T@X

        self.cov = np.triu(self.cov) + np.triu(self.cov, 1).T
        self.diag, self.coord = np.linalg.eigh(self.cov)
        self.diag = np.real(self.diag) + 10**-12

        return idx_sorted[-self.parents:]

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and sigma
        """
        return np.copy(self.mu), np.copy(self.cov)


class CMAES:

    """
    CMAES implementation adapted from
    https://en.wikipedia.org/wiki/CMA-ES#Example_code_in_MATLAB/Octave
    """

    def __init__(self,
                 num_params,
                 mu_init=None,
                 sigma_init=1,
                 step_size_init=1,
                 pop_size=255,
                 antithetic=False,
                 weight_decay=0.01,
                 rank_fitness=True):

        # distribution parameters
        self.num_params = num_params
        if mu_init is not None:
            self.mu = np.array(mu_init)
        else:
            self.mu = np.zeros(num_params)
        self.step_size = step_size_init
        self.coord = np.eye(num_params)
        self.diag = sigma_init ** 2 * np.ones(num_params)
        self.cov = sigma_init ** 2 * np.eye(num_params)
        self.inv_sqrt_cov = 1 / sigma_init * np.eye(num_params)
        self.p_c = np.zeros(self.num_params)
        self.p_s = np.zeros(self.num_params)
        self.antithetic = antithetic

        # selection parameters
        self.pop_size = pop_size
        self.parents = pop_size // 2
        self.weights = np.array([np.log((self.parents + 0.5) / i)
                                 for i in range(1, self.parents + 1)])
        self.weights /= self.weights.sum()
        self.parents_eff = 1 / (self.weights ** 2).sum()
        self.rank_fitness = rank_fitness
        self.weight_decay = weight_decay

        # adaptation  parameters
        self.c_s = (self.parents_eff + 2) / \
            (self.num_params + self.parents_eff + 5)
        self.c_c = (4 + self.parents_eff / self.num_params) / \
            (self.num_params + 4 + 2 * self.parents_eff / self.num_params)
        self.c_1 = 2 / ((self.num_params + 1.3) ** 2 + self.parents_eff)
        self.c_mu = min(1 - self.c_1, 2 * (self.parents_eff - 2 + 1 /
                                           self.parents_eff) / ((self.num_params + 2) ** 2
                                                                + self.parents_eff))
        self.damps = 1 + 2 * \
            max(0, np.sqrt((self.parents_eff - 1) /
                           (self.num_params + 1)) - 1) + self.c_s
        self.chi = np.sqrt(self.num_params) * \
            (1 - 1 / (4 * self.num_params) + 1 / (21 * self.num_params ** 2))
        self.count_eval = 0
        self.eigen_eval = 0

    def ask(self, pop_size):
        """
        Returns a list of candidates parameters
        """
        if self.antithetic:
            epsilon_half = np.random.randn(self.pop_size // 2, self.num_params)
            epsilon = np.concatenate([epsilon_half, - epsilon_half])

        else:
            epsilon = np.random.randn(self.num_params, pop_size)

        return self.step_size * (self.coord @ np.diag(np.sqrt(self.diag)) @ epsilon).T + self.mu

    def tell(self, solutions, scores):
        """
        Updates the distribution
        """
        # scores preprocess
        reward = np.array(scores)
        if self.rank_fitness:
            reward = compute_centered_ranks(reward)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, solutions)
            reward += l2_decay

        scores = -np.array(scores)
        idx_sorted = np.argsort(scores)

        # update mean
        old_mu = self.mu
        self.mu = self.weights @ solutions[idx_sorted[:self.parents]]

        # update evolution paths
        self.p_s = (1 - self.c_s) * self.p_s + \
            np.sqrt(self.c_s * (2 - self.c_s) * self.parents_eff) * \
            self.inv_sqrt_cov @ (self.mu - old_mu) / self.step_size

        tmp_1 = np.linalg.norm(self.p_s) / np.sqrt(self.c_s * (2 - self.c_s)) \
            <= self.chi * (1.4 + 2 / (self.num_params + 1))

        self.p_c = (1 - self.c_c) * self.p_c + \
            tmp_1 * np.sqrt(self.c_c * (2 - self.c_c) * self.parents_eff) * \
            (self.mu - old_mu) / self.step_size

        # update covariance matrix
        tmp_2 = 1 / self.step_size * \
            (solutions[idx_sorted[:self.parents]] - old_mu)

        self.cov = (1 - self.c_1 - self.c_mu) * self.cov + \
            (1 - tmp_1) * self.c_1 * self.c_c * (2 - self.c_c) * self.cov + \
            self.c_1 * np.outer(self.p_c, self.p_c) + \
            self.c_mu * tmp_2.T @ np.diag(self.weights) @ tmp_2

        # update step size
        self.step_size *= np.exp((self.c_s / self.damps) *
                                 (np.linalg.norm(self.p_s) / self.chi - 1))

        # decomposition of C
        self.cov = np.triu(self.cov) + np.triu(self.cov, 1).T
        self.diag, self.coord = np.linalg.eigh(self.cov)
        self.diag = np.real(self.diag)

        self.inv_sqrt_cov = self.coord @ np.diag(
            1 / np.sqrt(self.diag)) @ self.coord.T

        return idx_sorted[:self.parents]

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and the covariance matrix
        """
        return np.copy(self.mu), np.copy(self.step_size)**2 * np.copy(self.cov)

    def result(self):
        """
        Returns best params so far, best score, current score
        and sigma
        """
        pass


class sepCMAES:

    """
    CMAES implementation adapted from
    https://en.wikipedia.org/wiki/CMA-ES#Example_code_in_MATLAB/Octave
    """

    def __init__(self,
                 num_params,
                 mu_init=None,
                 sigma_init=1,
                 step_size_init=1,
                 pop_size=255,
                 antithetic=False):

        # distribution parameters
        self.num_params = num_params
        if mu_init is not None:
            self.mu = np.array(mu_init)
        else:
            self.mu = np.zeros(num_params)
        self.step_size = step_size_init
        self.diag = np.sqrt(sigma_init * np.ones(num_params))
        self.cov = sigma_init * np.ones(num_params)
        self.p_c = np.zeros(self.num_params)
        self.p_s = np.zeros(self.num_params)
        self.antithetic = antithetic

        # selection parameters
        self.pop_size = pop_size
        self.parents = pop_size // 2
        self.weights = np.array([np.log((self.parents + 0.5) / i)
                                 for i in range(1, self.parents + 1)])
        self.weights /= self.weights.sum()
        self.parents_eff = 1 / (self.weights ** 2).sum()

        # adaptation  parameters
        self.c_s = (self.parents_eff + 2) / \
            (self.num_params + self.parents_eff + 5)
        self.c_c = (4 + self.parents_eff / self.num_params) / \
            (self.num_params + 4 + 2 * self.parents_eff / self.num_params)
        self.c_1 = 2 / ((self.num_params + 1.3) ** 2 + self.parents_eff)
        self.c_mu = min(1 - self.c_1, 2 * (self.parents_eff - 2 + 1 /
                                           self.parents_eff) / ((self.num_params + 2) ** 2
                                                                + self.parents_eff))
        self.damps = 1 + 2 * \
            max(0, np.sqrt((self.parents_eff - 1) /
                           (self.num_params + 1)) - 1) + self.c_s
        self.chi = np.sqrt(self.num_params) * \
            (1 - 1 / (4 * self.num_params) + 1 / (21 * self.num_params ** 2))
        self.count_eval = 0
        self.eigen_eval = 0

    def ask(self, pop_size):
        """
        Returns a list of candidates parameters
        """
        epsilon = np.random.randn(pop_size, self.num_params)
        return self.mu + self.step_size * self.diag * epsilon

    def tell(self, solutions, scores):
        """
        Updates the distribution
        """
        scores = -np.array(scores)
        idx_sorted = np.argsort(scores)

        # update mean
        old_mu = self.mu
        self.mu = self.weights @ solutions[idx_sorted[:self.parents]]

        # update evolution paths
        self.p_s = (1 - self.c_s) * self.p_s + \
            np.sqrt(self.c_s * (2 - self.c_s) * self.parents_eff) * \
            1/self.diag * (self.mu - old_mu) / self.step_size

        tmp_1 = np.linalg.norm(self.p_s) / np.sqrt(self.c_s * (2 - self.c_s)) \
            <= self.chi * (1.4 + 2 / (self.num_params + 1))

        self.p_c = (1 - self.c_c) * self.p_c + \
            tmp_1 * np.sqrt(self.c_c * (2 - self.c_c) * self.parents_eff) * \
            (self.mu - old_mu) / self.step_size

        # update covariance matrix
        tmp_2 = 1 / self.step_size * \
            (solutions[idx_sorted[:self.parents]] - old_mu)

        self.cov = (1 - self.c_1 - self.c_mu) * self.cov + \
            (1 - tmp_1) * self.c_1 * self.c_c * (2 - self.c_c) * self.cov + \
            (self.num_params + 2)/3 * self.c_1 * np.outer(self.p_c, self.p_c) + \
            self.c_mu * tmp_2.T @ np.diag(self.weights) @ tmp_2

        # update step size
        self.step_size *= np.exp((self.c_s / self.damps) *
                                 (np.linalg.norm(self.p_s) / self.chi - 1))

        # decomposition of C
        self.diag = np.sqrt(np.diag(self.cov))

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and the covariance matrix
        """
        return self.mu, self.step_size * self.cov
