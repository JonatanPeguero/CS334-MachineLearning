'''THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS SUCH AS CHATGPT.
/* Jonatan Peguero  */'''

import numpy as np
import scipy
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

def random_argmax(arr):
    """
    Randomly select one of the indices with the maximum value
    This function helps to break ties randomly
    Input: 
        - arr: 1D numpy array
    Output:
        - index: int
    """
    return np.random.choice(np.where(arr == np.max(arr))[0])


class MultiArmedBandit(object):
    """
    Base class for multi-armed bandit problems
    """
    def __init__(self):
        # List of distributions for each arm
        self.R = []
        raise NotImplementedError
    
    def __len__(self):
        # Number of arms
        return len(self.R)

    def pull(self, arm):
        # Generate reward from the selected arm's distribution
        return self.R[arm].rvs()


class BernoulliMAB(MultiArmedBandit):
    """
    Multi-armed bandit with Bernoulli distributions
    """
    def __init__(self, means):
        self.means = np.array(means)
        self.R = [scipy.stats.bernoulli(p) for p in means]


def epsilon_greedy(mab, T, epsilon):
    # Initialization
    k = len(mab)
    Q = np.zeros(k)
    N = np.zeros(k)
    
    rewards = []  # reward at each time step
    optimals = [] # whether the optimal arm was selected at each time step
    regrets = []  # cumulative regret at each time step

    for t in range(T):

        # Action selection: epsilon-greedy
        if np.random.rand() < epsilon:
            a = np.random.choice(len(mab))
        else:
            a = random_argmax(Q)
        
        # Pull arm a, observe reward
        r = mab.pull(a)

        # Update N and Q
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        
        # Track learning metrics
        rewards.append(r)
        optimals.append(mab.means[a] == mab.means.max())
        regrets.append(mab.means.max() - mab.means[a])

    return Q, N, np.array(rewards), np.array(optimals), np.cumsum(regrets)


def optimistic_initialization(mab, T, Q0):
    # Initialization
    k = len(mab)
    Q = np.full(k, Q0, dtype=float)
    N = np.zeros(k)
    
    rewards = []  # reward at each time step
    optimals = [] # whether the optimal arm was selected at each time step
    regrets = []  # cumulative regret at each time step

    for t in range(T):
        # Action selection: greedy
        a = random_argmax(Q)
        
        # Pull arm a and observe reward
        r = mab.pull(a)

        # Update N and Q
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        
        # Track learning metrics
        rewards.append(r)
        optimals.append(mab.means[a] == mab.means.max())
        regrets.append(mab.means.max() - mab.means[a])

    return Q, N, np.array(rewards), np.array(optimals), np.cumsum(regrets)


def upper_confidence_bounds(mab, T, c):
    # Initialization
    k = len(mab)
    Q = np.zeros(k)
    N = np.zeros(k)
    
    rewards = []  # reward at each time step
    optimals = [] # whether the optimal arm was selected at each time step
    regrets = []  # cumulative regret at each time step

    for t in range(T):
        # Action selection: greedy wrt UCB scores
        bonus = c * np.sqrt(np.log(t+1) / (N + 1e-16))
        bonus[N == 0] = np.inf
        ucb_scores = Q + bonus
        a = random_argmax(ucb_scores)

        # Pull arm a and observe reward
        r = mab.pull(a)

        # Update N and Q
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        
        # Track learning metrics
        rewards.append(r)
        optimals.append(mab.means[a] == mab.means.max())
        regrets.append(mab.means.max() - mab.means[a])

    return Q, N, np.array(rewards), np.array(optimals), np.cumsum(regrets)


def plot_results(outputs, labels, colors, smooth=False):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for (rewards, optimals, regrets), label, color in zip(outputs, labels, colors):
        if smooth:
            # smooth the curves for reward and %optimal
            axes[0].plot(np.convolve(rewards, np.ones(10)/10, mode='valid'), lw=1, color=color, label=label)
            axes[1].plot(np.convolve(optimals, np.ones(10)/10, mode='valid'), lw=1, color=color, label=label)
            axes[2].plot(regrets, lw=1, color=color, label=label)
        else:
            axes[0].plot(rewards, lw=1, color=color, label=label)
            axes[1].plot(optimals, lw=1, color=color, label=label)
            axes[2].plot(regrets, lw=1, color=color, label=label)
    for ax, l, yrange in zip(axes, ['Reward at step t', '% Optimal Action at step t', 'Cumulative regret'], [(0, 1), (0,1), (0,T*0.12)]):
        ax.set_xlabel('timestep')
        ax.set_ylabel(l)
        ax.set_xlim(-T*0.05,T*1.05)
        ax.set_ylim(*yrange)
    fig.legend(*axes[0].get_legend_handles_labels(), bbox_to_anchor=(0., 1), loc='upper left', ncol=3)
    return fig, axes


def run_experiments_epsilon_greedy(mab, T, runs):
    eps_list = [0.02, 0.1, 0.3]

    # Run epsilon-greedy algorithm for different epsilon values
    all_outputs_1 = []
    for i, eps in enumerate(eps_list):
        results = []
        for run in tqdm(range(runs)):
            np.random.seed(run)
            random.seed(run)
            results.append(epsilon_greedy(mab, T=T, epsilon=eps))
        Q, n, rewards, optimals, regrets = zip(*results)
        rewards = np.array(rewards).mean(axis=0)
        optimals = np.array(optimals).mean(axis=0)
        regrets = np.array(regrets).mean(axis=0)
        all_outputs_1.append((rewards, optimals, regrets))
    
    # plot
    fig, axes = plot_results(
        all_outputs_1, 
        ['$\\epsilon={}$'.format(eps) for eps in eps_list], 
        ['r', 'b', 'g'], 
        smooth=True,
    )
    plt.suptitle('$\\epsilon$-greedy')
    plt.tight_layout()
    plt.savefig('bandit__epsilon_greedy.png', dpi=300, bbox_inches='tight')


def run_experiments_optimistic_init(mab, T, runs):
    Q0_list = [0, 1, 50]
    
    # Run optimistic initial values algorithm for different Q0 values
    all_outputs_2 = []
    for i, Q0 in enumerate(Q0_list):
        results = []
        for run in tqdm(range(runs)):
            np.random.seed(run)
            random.seed(run)
            results.append(optimistic_initialization(mab, T=T, Q0=Q0))
        Q, n, rewards, optimals, regrets = zip(*results)
        rewards = np.array(rewards).mean(axis=0)
        optimals = np.array(optimals).mean(axis=0)
        regrets = np.array(regrets).mean(axis=0)
        all_outputs_2.append((rewards, optimals, regrets))
    
    # plot
    fig, axes = plot_results(
        all_outputs_2,
        ['$Q_1={}$'.format(Q0) for Q0 in Q0_list],
        ['tab:gray', 'tab:cyan', 'tab:blue'],
        smooth=True,
    )
    plt.suptitle('Optimistic Initial Values')
    plt.tight_layout()
    plt.savefig('bandit__optimistic_initialization.png', dpi=300, bbox_inches='tight')


def run_experiments_ucb(mab, T, runs):
    c_list = [0.2, 1.0, 2.0]

    # Run UCB algorithm for different c values
    all_outputs_3 = []
    for i, c in enumerate(c_list):
        results = []
        for run in tqdm(range(runs)):
            np.random.seed(run)
            random.seed(run)
            results.append(upper_confidence_bounds(mab, T=T, c=c))
        Q, n, rewards, optimals, regrets = zip(*results)
        rewards = np.array(rewards).mean(axis=0)
        optimals = np.array(optimals).mean(axis=0)
        regrets = np.array(regrets).mean(axis=0)
        all_outputs_3.append((rewards, optimals, regrets))
    
    # plot
    fig, axes = plot_results(
        all_outputs_3,
        ['$c={}$'.format(c) for c in c_list],
        ['tab:pink', 'purple', 'm'],
        smooth=True,
    )
    plt.suptitle('Upper Confidence Bounds')
    plt.tight_layout()
    plt.savefig('bandit__upper_confidence_bounds.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Define the problem
    means = [0.1, 0.275, 0.45, 0.625, 0.8]
    np.random.shuffle(means)
    mab = BernoulliMAB(means)
    T = 5000
    runs = 100

    # Run the algorithms
    run_experiments_epsilon_greedy(mab, T, runs)
    run_experiments_optimistic_init(mab, T, runs)
    run_experiments_ucb(mab, T, runs)
