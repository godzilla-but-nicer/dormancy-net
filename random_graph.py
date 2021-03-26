#%% [markdown]
# # Erdos-Renyi Random Graph
# This notebook contains the code needed to generate random graphs as described
# by Erdos and Renyi in like the 60s or whatever.
#
# Let $N$ be the number of nodes in a set of nodes $n$. We will construct a
# graph $G = \{n, E\}$ where $E$ is a set of edges. We will construct $E$ and
# with it our graph $G$ by connecting a pair of nodes $n_i, n_j \in n$ with
# some probability $p$ for all possible pairs of nodes.
# 
# We're going to start by importing the packages we need and setting up some 
# objects that we're going to want to use.
# %%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# seed our random numbers
rng = np.random.default_rng(1234)

# %% [markdown]
# Ok now we will set up the algorithm we will use to
# construct our graph. We're going to use a simple non-optimal algorithm for
# illustrative purposes.
# %%
def ER_graph(N, p):
    G = nx.Graph()
    for n_i in range(N):
        for n_j in range(n_i + 1, N):
            if rng.uniform() < p:
                G.add_edge(n_i, n_j)
    
    return G

# %% [markdown]
# This generative process is very well studied. as $N \rightarrow \infty$ 
# and the product $Np$ stays constant the
# degree distribution approaches a poisson distribution. We can use this
# property to check our implementation. Let's implement a poisson distribution.
# %%
def poisson_pdf(k, l):
    return (l**k * np.exp(-l)) / np.math.factorial(k)
# %% [markdown]
# Ok let's get into it. We can't actually do $N \rightarrow \infty$ but we can
# take a look at big-ish N.
# %%
# set parameters
N = 1000
p = 0.01

# generate the graph
G = ER_graph(N, p)

# get the degree distribution
deg = nx.degree_histogram(G)

# get the theoretical distribution
deg_range = np.arange(0, len(deg) + 1)
ppdf = [poisson_pdf(k, N*p) for k in deg_range]

plt.bar(deg_range[:-1], deg, label='Obs.')
plt.plot(deg_range, ppdf, label='Theo.', c='C1')
plt.legend()
plt.show()
# %%
