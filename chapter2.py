import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    def __init__(self, K : int) -> None:
        self.probs = np.random.uniform(size = K)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.K = K
    
    def step(self, k : int):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0

class Solver:
    def __init__(self, bandit : BernoulliBandit) -> None:
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0
        self.actions = []
        self.regrets = []
    
    def update_regret(self, k):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        raise NotImplementedError
    
    def run(self, num_steps):
        for i in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

class EpsilonGreedy(Solver):
    def __init__(self, bandit: BernoulliBandit, epsilon = 0.01, init_prob=1.0) -> None:
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.K)
    
    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

class DecayingEpsilonGreedy(Solver):
    def __init__(self, bandit: BernoulliBandit, init_prob : float = 1.0) -> None:
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1/self.total_count:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

class ubc(Solver):
    def __init__(self, bandit: BernoulliBandit, coef, init_prob=1.0) -> None:
        super(ubc, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef
        
    def run_one_step(self):
        self.total_count += 1
        ubckk = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1))
        )
        k = np.argmax(ubckk)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

class ThompsonSampling(Solver):
    def __init__(self, bandit: BernoulliBandit) -> None:
        super(ThompsonSampling, self).__init__(bandit)
        self.total_count = 0
        self._a = np.ones(self.bandit.K)
        self._b = np.ones(self.bandit.K)
        
    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)
        k = np.argmax(samples)
        r = self.bandit.step(k)

        self._a[k] += r
        self._b[k] += 1 - r
        return k

def plot_results(solvers: Solver, solver_names):
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label = solver_names[idx])
    plt.xlabel("Time steps")
    plt.ylabel("Cumulative regrets")
    plt.title("%d-armed bandit" % solvers[0].bandit.K)
    plt.legend()
    plt.show()
    plt.close()

if __name__ == "__main__":
    np.random.seed(0)
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
    epsilon_greedy_solver.run(6000)
    plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])

    epsilons = [0.0001, 0.01, 0.1, 0.26, 0.6]
    epsilon_greedy_solver_list = [
        EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons
    ]
    epsilon_greedy_solver_name = ["epsilon={}".format(e) for e in epsilons]
    for solver in epsilon_greedy_solver_list:
        solver.run(600)
    plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_name)
    
    decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
    decaying_epsilon_greedy_solver.run(6000)
    plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])

    coef = 1.0
    ubc_solver = ubc(bandit_10_arm, coef)
    ubc_solver.run(6000)
    plot_results([ubc_solver], ["ubc"])
    
    thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
    thompson_sampling_solver.run(6000)
    plot_results([thompson_sampling_solver], ["ThompsonSampling"])