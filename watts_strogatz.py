#%% [markdown]
# # Watts-Strogatz Small-World Network
# 
# In 1999, Duncan Watts and Steve Strogatz proposed a generative network model
# somewhere in between an ordered lattice and a random graph. Starting from a
# regular lattice consisting of $N$ nodes, each connected to their $z$ nearest 
# neighbors. We iterate over the edges and with probability $p$ we "rewire" the
# edge connecting one of its ends to a random node. The model produces short
# path lengths and high clustering which are characteristics of real networks
# that the ER random graph can't produce.
#
# Let's import the packages we'll need

#%%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# seed our random number generator
rng = np.random.default_rng(1234)

#%% [markdown]
# First, we'll implement the original Watts-Strogatz model as a function
def watts_strogatz(N, z, p):
    G = nx.circulant_graph(N, range(1, z+1))

    # will be useful to have some stuff
    node_labs = np.arange(N)
    edge_set = set(G.edges())

    # ok let's check each edge for rewiring
    edges = G.edges()
    for e in edges:
        if rng.uniform() < p:
            # no self edges
            others = node_labs[node_labs != e[0]]

            # no duplicate edges
            for _ in others:
                target = rng.choice(others)
                new_edge = (e[0], target)

                if new_edge not in edge_set:
                    G.remove_edge(e[0], e[1])
                    G.add_edge(e[0], target)
                    break
            
            # remake the edge set (maybe not efficient)
            edge_set = set(G.edges())

    return G

G = watts_strogatz(25, 5, 0.002)
nx.draw_circular(G, node_size=50)
# %% [markdown]
# Looks like a small-world network! We'll plot the growth of path lengths and
# the clustering coefficient to be sure we're geting the characteristic results.
# %%
net_sizes = np.arange(100, 2100, 100)
path_lengths = []
clustering = []
for N in net_sizes:
    G = watts_strogatz(N, 5, 0.02)
    path_lengths.append(nx.average_shortest_path_length(G))
    clustering.append(nx.average_clustering(G))

fig, ax = plt.subplots(ncols=2)
ax[0].plot(net_sizes, np.log(net_sizes))
ax[0].scatter(net_sizes, path_lengths)
ax[0].set_xlabel('N')
ax[0].set_ylabel('Average Path Length')

ax[1].scatter(net_sizes, clustering)
ax[1].set_xlabel('N')
ax[1].set_ylabel('Average Clustering')
plt.tight_layout()
plt.show()
# %% [markdown]
# OK that looks right!
# %%
