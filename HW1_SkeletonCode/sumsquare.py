#THIS HOMEWORK IS MY OWN WORK, WRITTEN WITHOUT COPYING FROM OTHER STUDENTS
#OR DIRECTLY FROM LARGE LANGUAGE MODELS SUCH AS CHATGPT.
#Any collaboration or external resources have been properly acknowledged.
#Jonatan Peguero
"""
Vectorization Comparison for Computing Sum of Squares
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""
import numpy as np  
import timeit
import pandas as pd
import matplotlib.pyplot as plt
def gen_random_samples(n):
    """
    Generate n random samples using the
    numpy random.randn module.

    Returns
    ----------
    sample : 1d array of size n
        An array of n random samples
    """
    # TODO: Implement this function
    return np.random.randn(n)


def sum_squares_for(samples):
    """
    Compute the sum of squares using a forloop

    Parameters
    ----------
    samples : 1d-array with shape n
        An array of numbers.

    Returns
    -------
    ss : float
        The sum of squares of the samples
    """
    # TODO: Implement this function
    ss = 0.0  
    for sample in samples:
        ss += sample ** 2  
    return ss


def sum_squares_np(samples):
    """
    Compute the sum of squares using Numpy's dot module

    Parameters
    ----------
    samples : 1d-array with shape n
        An array of numbers.

    Returns
    -------
    ss : float
        The sum of squares of the samples
    """
    # TODO: Implement this function
    return np.dot(samples, samples)


def time_ss(sample_list):
    """
    Time it takes to compute the sum of squares
    for varying number of samples. The function should
    generate a random sample of length s (where s is an 
    element in sample_list), and then time the same random 
    sample using the for and numpy loops.

    Parameters
    ----------
    samples : list of length n
        A list of integers to .

    Returns
    -------
    ss_dict : Python dictionary with 3 keys: n, ssfor, ssnp.
        The value for each key should be a list, where the 
        ordering of the list follows the sample_list order 
        and the timing in seconds associated with that 
        number of samples.
    """
    # TODO: Implement this function
    ss_dict = {'n': [], 'ssfor': [], 'ssnp': []}
    
    for s in sample_list:
        samples = gen_random_samples(s)
        start = timeit.default_timer()
        sum_squares_for(samples)
        elapsed_for = timeit.default_timer() - start
        start = timeit.default_timer()
        sum_squares_np(samples)
        elapsed_np = timeit.default_timer() - start
        ss_dict['n'].append(s)
        ss_dict['ssfor'].append(elapsed_for)
        ss_dict['ssnp'].append(elapsed_np)
    
    return ss_dict


def timess_to_df(ss_dict):
    """
    Time the time it takes to compute the sum of squares
    for varying number of samples.

    Parameters
    ----------
    ss_dict : Python dictionary with 3 keys: n, ssfor, ssnp.
        The value for each key should be a list, where the 
        ordering of the list follows the sample_list order 
        and the timing in seconds associated with that 
        number of samples.

    Returns
    -------
    time_df : Pandas dataframe that has n rows and 3 columns.
        The column names must be n, ssfor, ssnp and follow that order.
        ssfor and ssnp should contain the time in seconds.
    """
    # TODO: Implement this function
    time_df = pd.DataFrame(ss_dict, columns=['n', 'ssfor', 'ssnp'])
    return time_df

def plot_timings(time_df):
    plt.figure(figsize=(12, 8))
    plt.plot(time_df['n'], time_df['ssfor'], label='sum_squares_for', marker='o', linestyle='-', color='blue')
    plt.plot(time_df['n'], time_df['ssnp'], label='sum_squares_np', marker='x', linestyle='--', color='red')
    plt.xlabel('Number of Samples (n)', fontsize=14)
    plt.ylabel('Execution Time (seconds)', fontsize=14)
    plt.title('Performance Comparison: For-Loop vs. NumPy Dot', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.xscale('log')  
    plt.yscale('log') 
    plt.tight_layout()
    plt.show()

def main():
    # generate 100 samples
    samples = gen_random_samples(100)
    #print(samples)
    # call the for version
    ss_for = sum_squares_for(samples)
    #print(ss_for)
    # call the numpy version
    ss_np = sum_squares_np(samples)
    #print(ss_np)
    # make sure they are approximately the same value
    sample_sizes = [1, 50, 100, 1000, 10000, 100000]
    ss_timings = time_ss(sample_sizes)
    #print(ss_timings)
    time_df = timess_to_df(ss_timings)
    #print(time_df)
    plot_timings(time_df)
    import numpy.testing as npt
    npt.assert_almost_equal(ss_for, ss_np, decimal=5)


if __name__ == "__main__":
    main()
