#%%
import numpy as np
import pandas as pd
import networkx as nx

def toggle_n(n:int, Self=None):
    """
    Generates a fully connected directed graph with n nodes and sets the mutual repression between nodes.
    If Self is provided as 'SA' or 'SI', it sets self-loops as activation or inhibition, respectively.
    
    Parameters:
    - n (int): Number of nodes in the graph.
    - Self (str, optional): Type of self-loops. Default is None.
    
    Returns:
    - G (pd.DataFrame): Pandas DataFrame representing the graph.

    Saves:
    - ./TOPO/T{n}.topo (File): Topology file of the graph.
    """
    if n<=1:
        raise ValueError('n must be greater than 1')
    # Get a fully connected directed graph
    G = nx.complete_graph(n, create_using=nx.DiGraph)
    # Set the node mapping to alphabets
    mapping = { i:chr(65+i) for i in range(n) }
    G = nx.relabel_nodes(G, mapping)
    # Set the mutual repression
    for edge in G.edges:
        G.edges[edge]['Action'] = 2
    if Self:
        if Self=='SA':
            selfweight = 1
        elif Self=='SI':
            selfweight = 2
        else:
            raise ValueError('Self must be either SA or SI')
        # Set the self loops as activation or inhibition
        for node in G.nodes:
            G.edges[(node,node)]['Action'] = selfweight
    # Convert to topofile
    G = nx.to_pandas_edgelist(G)
    G.to_csv('./TOPO/T'+str(n)+Self+'.topo', sep='\t', index=False)
    return G

def team_n(n:int, m:int):
    """
    Generates a directed graph representing a team structure with n teams and m members per team.
    Each team has intra-team activations and inter-team inhibitions.
    
    Parameters:
    - n (int): Number of teams.
    - m (int): Number of members per team.
    
    Returns:
    - G (pd.DataFrame): Pandas DataFrame representing the graph.

    Saves:
    - ./TOPO/Team{n}_{m}.topo (File): Topology file of the graph.
    """
    if (n<=1) or (m<=1):
        raise ValueError('n and m must be greater than 1')
    # Make a adjacency matrix with all inhibitions
    A = np.full((n*m,n*m), 2)
    for i in range(n):
        # Set the intra team activations
        A[i*m:(i+1)*m,i*m:(i+1)*m] = 1
    # Set no self loops
    np.fill_diagonal(A, 0)
    # Convert to networkx graph
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    # Set the node mapping to alphabets
    mapping = { i:chr(65+i//m)+str(i%m+1) for i in range(n*m) }
    G = nx.relabel_nodes(G, mapping)
    # Convert to topofile
    G = nx.to_pandas_edgelist(G)
    G.to_csv('./TOPO/Team'+str(n)+'_'+str(m)+'.topo', sep='\t', index=False)
    return G