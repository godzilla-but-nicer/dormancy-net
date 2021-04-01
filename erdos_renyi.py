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
def erdos_renyi(N, p):
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
# %% [markdown]
# Ok let's get into it. We can't actually do $N \rightarrow \infty$ but we can
# take a look at big-ish N.
# %%
# set parameters
N = 1000
p = 0.005

# generate the graph
G = erdos_renyi(N, p)

# get the degree distribution
deg = nx.degree_histogram(G)
deg_probs = deg / np.sum(deg)

# get the theoretical distribution
def poisson_pdf(k, l):
    return (l**k * np.exp(-l)) / np.math.factorial(k)
deg_range = np.arange(0, len(deg) + 1)
ppdf = [poisson_pdf(k, N*p) for k in deg_range]

plt.bar(deg_range[:-1], deg_probs, label='Obs.')
plt.plot(deg_range, ppdf, label='Theo.', c='C1')
plt.legend()
plt.show()
# %% [markdown]
# Degree values for graphs with $N \approx 1000$ or bigger look like they are
# poisson distributed. So thats good, I think our implementation works. 
#
#%% [markdown]
## Adding dormancy at an individual level
# What we're really here to do is add dormancy to these models. We'll start
# with a population inspired model thinking of nodes as individuals and edges
# as some kind of interaction between them. Note that 
# network models in ecology usually describe communities with nodes as species
# which is not what we are doing here.
#
# Let $d_i$ be the probability that individual $i$ enters dormancy from the
# active state and $r_i$ be the probability that individual $i$ enters the
# active state from dormancy. We will also introduce another parameter, $f$ as 
# the fraction of individuals capable of dormancy.
#
# Our process will work like this for each 'timestep' where we check an edge:
# 
# 1. Update the states of the nodes
# 1. Check the state (Active or dormant) of each node in the pair
# 1. If both are active draw an edge with probability $p$ as in the original ER 
# process
#
# We'll start with all individual parameters equal $d_i = d$ and $r_i = r$.
# %%
def ER_dormancy(N, p, d, r, f):
    G = nx.Graph()
    for n in range(N):
        # add each node if they aren't already there. only need to do this for th
        if rng.uniform() < f:
            G.add_node(n, d=d, r=r, active=True)
        else:
            G.add_node(n, d=0, r=0, active=True)
        
    for n_i in range(N):
        for n_j in range(n_i + 1, N):
            # 1. update states
            if G.nodes[n_i]['active']:
                if rng.uniform() < G.nodes[n_i]['d']:
                    G.nodes[n_i]['active'] = False
            else:
                if rng.uniform() < G.nodes[n_i]['r']:
                    G.nodes[n_i]['active'] = True
            if G.nodes[n_j]['active']:
                if rng.uniform() < G.nodes[n_j]['d']:
                    G.nodes[n_j]['active'] = False
            else:
                if rng.uniform() < G.nodes[n_j]['r']:
                    G.nodes[n_j]['active'] = True
            # 2. check states
            if G.nodes[n_i]['active'] and G.nodes[n_j]['active']:
                # 3. check connection
                if rng.uniform() < p:
                    G.add_edge(n_i, n_j)
    
    return G

G = ER_dormancy(1000, 0.005, 0, 0, 0)
nx.draw(G)
# %% [markdown]
# set parameters
N = 3000
p = 0.01
d = 0.05
r = 0.02
f = 0.7

# generate the graph
G = ER_dormancy(N, p, d, r, f)

# get the degree distribution
deg = nx.degree_histogram(G)
deg_probs = deg / np.sum(deg)

# get the theoretical distribution without dormancy
deg_range = np.arange(0, len(deg) + 1)
ppdf = [poisson_pdf(k, N*p) for k in deg_range]

# now we'll also get the theoretical distribution with dormancy
def dormant_pdf(x, N, p, d, r, f):
    q = d / (d + r)
    no_dormancy_frac = (1 - f) * poisson_pdf(x, N * p * (1 - q * f))
    dormancy_frac = f * poisson_pdf(x, N * p * (1 - q * f) * (1 - q))
    return dormancy_frac + no_dormancy_frac

dpdf = [dormant_pdf(x, N, p, d, r, f) for x in deg_range]


plt.bar(deg_range[:-1], deg_probs, label='Obs.', alpha=0.4)
# plt.plot(deg_range, ppdf, label='Poisson', c='C1')
plt.plot(deg_range, dpdf, label='Theo.', c='C2', linestyle='--')
plt.xlabel('k')
plt.ylabel('P(k)')
plt.legend()
plt.show()
# %%
