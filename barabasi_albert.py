# %% [markdown]
# # Barabasi and Albert - Preferrential Attachment
#
# Preferential attachment had been given many names in many contexts before
# Barabasi and Albert's landmark 1999 paper. Still, the impact of this paper
# and the model it presented is hard to overstate.
#
# In preferrential attachment, a network grows from an initial set of $m_0$ 
# nodes by the addition of one node at a time. Each incoming node connects to 
# $m$ nodes in the network. The catch is that the incoming nodes don't connect
# uniformly, but instead connects to established nodes with 
# probability proportional to their degree.
# %%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# seed random number generator
rng = np.random.default_rng(1234)
#%% [markdown]
# Now let's implement the model.
# %%
# we'll do m_0 to simplify the number of parametersa
def barabasi_albert(N, m):
    G = nx.Graph()
    
    # start with the initial m nodes and add the next node
    G.add_nodes_from(range(m))
    init_edges = [(i, m+1) for i in range(m)]
    G.add_edges_from(init_edges)

    # now add the rest
    for i in range(m + 2, N):
        # connect with probability proportional to degree
        degrees = np.array([d[1] for d in G.degree()])
        pa_probs = degrees / np.sum(degrees)
        # make connections
        targets = rng.choice(range(degrees.shape[0]), p=pa_probs, size=m, replace=False)
        new_edges = [(i, t) for t in targets]
        G.add_edges_from(new_edges)

    return G

G = barabasi_albert(50, 5)
nx.draw_spring(G)
# %% [markdown]
# The goal of this model was to produce the power law degree distribution
# observed for many empirical networks so we'll check our implementation 
# using its degree distribution. People usually do a CCDF for these kinds of
# distributions but we'll just go with the classic PDF like they do in the paper
# %%
m = 3
G = barabasi_albert(1000, m)
degrees = [d[1] for d in G.degree()]
sorted_degree = sorted(degrees)

# calculate bins
n_bins = max(10, round(np.unique(sorted_degree).shape[0]/2))
bins = np.logspace(np.log10(sorted_degree[0]), np.log10(sorted_degree[-1]), n_bins, base=10)
counts, bin_edges = np.histogram(sorted_degree, bins, density=True)
norm_counts = np.array(counts) / np.diff(np.array(bin_edges))

# count CCDF
ccdf = np.cumsum(norm_counts[::-1])
rev_bins = bin_edges[::-1]

# get the theoretical power-law line as given in the paper
deg_lims = np.unique(sorted_degree)
pl_theo = 2 * m / deg_lims**(4)

plt.plot(rev_bins[1:], ccdf, label='Obs.')
plt.plot(deg_lims, pl_theo, linestyle='--', c='C1', label='Theo.')
plt.xlabel('k')
plt.ylabel('P(k)')
plt.xscale('log')
plt.yscale('log')
plt.show()
# %% [markdown]
# Looks good! Now that our BA model is working we can think about the role of
# dormancy. We'll start by implementing the model from Wang *et al.*
# we're going to add nodes one at a time like in BA model. These nodes will
# attach with preferrential attachment. but we're going to put to sleep and 
# wake up nodes at each time step. 
# 
# The process looks like this:
# 
# 1. Add new node with m connections [maybe with preferrential attachment?] only to active nodes
# 1. Set incoming node state to active
# 1. Randomly wake up a node
# 1. Set two nodes to the dormant state with probability:
#  $$ \nu_j = \frac{\gamma - 1}{\alpha + k^{in}_j} $$
#   where,
#  $$ \gamma - 1 = \left [ \sum_{l \in A} \frac{1}{\alpha + k^{in}_l} \right ]^{-1} $$
#
# so yeah let's try and implement this thing
# %%
def BA_dormancy(N, m, alpha):
    qfn = 2
    G = nx.DiGraph()
    
    # start with the initial m nodes and add the next node
    G.add_nodes_from(range(m), dormant=False)
    init_edges = [(i, m+1) for i in range(m)]
    G.add_edges_from(init_edges)
    # need dormancy vector for downstream operations
    ddict = nx.get_node_attributes(G, 'dormant')
    dormant = np.array([ddict[i] for i in len(ddict)])

    # now add the rest
    for i in range(m + 2, N):
        # connect randomly if node is not dormant, 
        targets = rng.choice(np.where(dormant == False), size=m, replace=False)
        new_edges = [(i, t) for t in targets]
    
        # wake up one dormant node
        wake_up = rng.choice(np.where(dormant), size=qfn - 1, replace=False)

        # make dormant two nodes according to probability above


        indeg = np.array([d[1] for d in G.in_degree()])
        pa_probs = indeg / np.sum(indeg)
        G.add_edges_from(new_edges)

    return G
G = BA_dormancy(50, 5, 10)
nx.draw_spring(G)
# %%

# %%
