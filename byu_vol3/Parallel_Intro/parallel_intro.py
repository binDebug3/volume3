# iPyParallel - Intro to Parallel Programming
from mpi4py import MPI
from ipyparallel import Client
import time
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt

# Problem 1
def prob1():
    """
    Write a function that initializes a Client object, creates a Direct
    View with all available engines, and imports scipy.sparse as sparse on
    all engines. Return the DirectView.
    """
    # initialize client
    client = Client()
    client.ids
    dview = client[:]

    # import scipy.sparse as sparse
    dview.execute('import scipy.sparse as sparse')

    # close client and return
    client.close()
    return dview


# Problem 2
def variables(dx):
    """
    Write a function variables(dx) that accepts a dictionary of variables. Create
    a Client object and a DirectView and distribute the variables. Pull the variables back and
    make sure they haven't changed. Remember to include blocking.
    """
    # initialize client
    client = Client()
    dview = client[:]
    dview.block = True

    # distribute and gather variables
    dview.push(dx)
    for d in dx.keys():
        dview.pull(d)

    # close client and return
    client.close()
    return dview


# Problem 3
def prob3(n=1000000):
    """
    Write a function that accepts an integer n.
    Instruct each engine to make n draws from the standard normal
    distribution, then hand back the mean, minimum, and maximum draws
    to the client. Return the results in three lists.
    
    Parameters:
        n (int): number of draws to make
        
    Returns:
        means (list of float): the mean draws of each engine
        mins (list of float): the minimum draws of each engine
        maxs (list of float): the maximum draws of each engine.
    """
    # initialize client
    client = Client()
    dview = client[:]
    dview.block = True

    # execute commands
    dview.execute('import numpy as np')
    dview.execute('np.random.seed(25)')
    dview.execute('draws = np.random.normal(size={})'.format(n))
    dview.execute('mean = draws.mean()')
    dview.execute('min = draws.min()')
    dview.execute('max = draws.max()')

    # pull results
    means = dview.pull('mean')
    mins = dview.pull('min')
    maxs = dview.pull('max')

    # close client and return
    client.close()
    return means, mins, maxs


# Problem 4
def prob4():
    """
    Time the process from the previous problem in parallel and serially for
    n = 1000000, 5000000, 10000000, and 15000000. To time in parallel, use
    your function from problem 3 . To time the process serially, run the drawing
    function in a for loop N times, where N is the number of engines on your machine.
    Plot the execution times against n.
    """
    n_list = [1000000, 5000000, 10000000, 15000000]
    time_parallel = []
    time_serial = []
    N = 4
    np.random.seed(25)

    for n in n_list:
        start = time.perf_counter()
        prob3(n)
        time_parallel.append(time.perf_counter() - start)

        start = time.perf_counter()
        for i in range(N):
            draws = np.random.normal(size=n)
            mean = draws.mean()
            min = draws.min()
            max = draws.max()
        time_serial.append(time.perf_counter() - start)
    
    plt.plot(n_list, time_parallel, label='Parallel')
    plt.plot(n_list, time_serial, label='Serial')
    plt.xlabel('n')
    plt.ylabel('time')
    plt.title('Time to Draw n Draws from Normal Distribution')
    plt.legend()
    plt.show()

    # save the plot
    # plt.savefig('/mnt/c/Users/dalli/source/acme_senior/vl3labs/Parallel_Intro/time_plot.png')


# Problem 5
def parallel_trapezoidal_rule(f, a, b, n=200):
    """
    Write a function that accepts a function handle, f, bounds of integration,
    a and b, and a number of points to use, n. Split the interval of
    integration among all available processors and use the trapezoidal
    rule to numerically evaluate the integral over the interval [a,b].

    Parameters:
        f (function handle): the function to evaluate
        a (float): the lower bound of integration
        b (float): the upper bound of integration
        n (int): the number of points to use; defaults to 200
    Returns:
        value (float): the approximate integral calculated by the
            trapezoidal rule
    """
    # initialize client
    client = Client()
    dview = client[:]
    dview.block = True

    # build domain
    x = np.linspace(a, b, n+1)
    dview.execute('import numpy as np')

    # calculate integral
    value = dview.map(lambda k0, k1: np.sum(k0 + k1), f(x[:-1]), f(x[1:]))
    value = np.sum(value) * (x[1] - x[0])/(2)

    # close client and return
    client.close()
    return value



if __name__ == '__main__':
    # Test problem 1
    # dview = prob1()

    # # test problem 2
    # dxx = {'a': 1, 'b': 2, 'c': 3}
    # dview = variables(dxx)

    # # test problem 3
    # means, mins, maxs = prob3()
    # print(means)
    # print(mins)
    # print(maxs)

    # # test problem 4
    # prob4()

    # # test problem 5
    # f = lambda x: x**2
    # a = 0
    # b = 1
    # value = parallel_trapezoidal_rule(f, a, b, n=200)
    # print(value)