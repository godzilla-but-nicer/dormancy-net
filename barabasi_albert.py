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
        targets = rng.choice(range(degrees.shape[0]), p=pa_probs, size=m)
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
G = barabasi_albert(10000, m)
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
# %%
