import numpy as np
import pandas as pd
import networkx as nx
import itertools as it

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
            G.add_edge(node, node, Action=selfweight)
    else:
        Self = ''
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

def team_n_rand_split(n:int, m_tot:int):
    """
    Generates a directed graph representing a team structure with n teams and m_tot total members split randomly.
    Each team has intra-team activations and inter-team inhibitions.
    
    Parameters:
    - n (int): Number of teams.
    - m_tot (int): Total number of members across all teams.
    - n_nets (int): Number of random networks to generate.
    
    Returns:
    - G (pd.DataFrame): Pandas DataFrame representing the graph.

    Saves:
    - ./TOPO/Team{n}_{m_i}.topo (File): Topology file of the graph.
    """
    if (n<=1) or (m_tot<=n):
        raise ValueError('n must be greater than 1 and m_tot must be greater than n')
    # Make a adjacency matrix with all inhibitions
    A = np.full((m_tot,m_tot), 2)
    mapping = {}
    # Split m_tot into n groups
    splits = np.zeros(n)
    # Make sure no team is empty or all teams are equal
    while (0 in splits) or (len(set(splits))==1):
        splits = np.random.multinomial(m_tot, [1/n]*n)
    for i in range(n):
        # Set the intra team activations
        start = splits[:i].sum()
        end = start+splits[i]
        A[start:end,start:end] = 1
        mapping.update({ j:chr(65+i)+str(j-start+1) for j in range(start,end) })
    # Set no self loops
    np.fill_diagonal(A, 0)
    # Convert to networkx graph
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    # Set the node mapping to alphabets for 
    G = nx.relabel_nodes(G, mapping)
    # Convert to topofile
    G = nx.to_pandas_edgelist(G)
    fname = './TOPO/Team'+str(n)+'_'+'_'.join([str(i) for i in splits])+'.topo'
    G.to_csv(fname, sep='\t', index=False)
    return G

def embedded(G_t:nx.classes.digraph.DiGraph, size:int, density:float, nnets:int, core_name:str):
    if (size<=1):
        raise ValueError('size must be greater than 1')
    n = G_t.number_of_nodes()
    for i in range(nnets):
        # Generate a random graph
        G_r = nx.gnm_random_graph(size, round(density*size), directed=True)
        # Add edge weights to the random graph
        act = np.random.randint(1,3,len(list(G_r.edges)))
        nx.set_edge_attributes(G_r, dict(zip(G_r.edges, act)), 'weight')
        # Add the toggle_n graph to the random graph 
        G = nx.compose(G_t, G_r)
        # Generate (n x size) randint matrix for the connections between the two graphs
        G_c = pd.DataFrame(0,columns=G.nodes, index=G.nodes)
        G_c.loc[list(G_r.nodes), list(G_t.nodes)] = np.random.randint(0,3,(size,n))
        G_c.loc[list(G_t.nodes), list(G_r.nodes)] = np.random.randint(0,3,(n,size))
        # Convert to networkx graph       
        G_c = nx.from_pandas_adjacency(G_c, create_using=nx.DiGraph)
        # Add the connections to the graph
        G = nx.compose(G, G_c)
        # Save the graph topofile
        fname = f'./TOPO/Embedded_{core_name}_{size}_{density}_{i}.topo'
        df = nx.to_pandas_edgelist(G)
        df.to_csv(fname, sep='\t', index=False)

def embedded_toggle(n:int, size:int, density:float, nnets:int, Self=None):
    """
    Generates directed graphs of toggle_n embedded in a random graphs.

    Parameters:
    - n (int): Number of nodes in the graph.
    - size (int): Number of nodes in the random graph.
    - density (float): Density (avg. number of edges per node) of the random graph.
    - nnets (int): Number of random networks to generate.
    - Self (str, optional): Type of self-loops. Default is None.

    Returns:
    - None

    Saves:
    - ./TOPO/Embedded_T{n}_{size}_{density}_{i}.topo (File): Topology file of the graph.
    """
    if (n<=1) or (size<=1):
        raise ValueError('n and size must be greater than 1')
    # Generate the toggle_n graph
    G_t = toggle_n(n, Self)
    G_t.columns = ['source', 'target', 'weight']
    G_t = nx.from_pandas_edgelist(G_t, source='source', target='target', edge_attr='weight', create_using=nx.DiGraph)
    embedded(G_t, size, density, nnets, f'T{n}')    

def impure_n(n:int, m:int, nnets:int):
    """
    Generates fully connected directed graphs with n nodes and m impurities (edges with opposite sign).

    Parameters:
    - n (int): Number of nodes in the graph.
    - m (int): Number of Impure edges.
    - nnets (int): Number of random networks to generate.

    Returns:
    - None

    Saves:
    - ./TOPO/Impure_T{n}_{imp}_{i}.topo (File): Topology file of the graph.
    """
    if (n<=1) or (m<0) or (m>n*(n-1)):
        raise ValueError('n must be greater than 1; m must be greater than 0 and less than n*(n-1)')
    # Initialize empty n node graph
    G = nx.DiGraph()
    G.add_nodes_from(range(0,n))
    # Pandas edgelist of complete graph
    df_c = nx.to_pandas_edgelist(nx.complete_graph(n,nx.DiGraph()))
    df_c['type']=2
    # generate all possible pairs of nodes
    node_pairs = list(it.product(range(0, n), repeat=2))
    # filter out self-loops
    node_pairs = list(filter(lambda x: x[0] != x[1], node_pairs))
    # create all possible permutations of m edges
    edge_permutations = list(it.combinations(node_pairs, m))
    np.random.shuffle(edge_permutations)
    # create a list of graphs with each permutation of edges
    g_uniq = []
    g_n = 0
    for edges in edge_permutations:
        G1 = G.copy()
        G1.add_edges_from(edges)
        for G2 in g_uniq:
            # if the graph is isomorphic to existing do nothing
            if nx.algorithms.is_isomorphic(G1, G2):
                break
        else:
            # or add it to unique list
            g_uniq.append(G1)
            # Convert the Graph to dataframe
            df_g = nx.to_pandas_edgelist(G1)
            df = df_c.copy()
            # ????
            keys = list(df_g.columns.values)
            i1 = df[keys].set_index(keys).index
            i2 = df_g.set_index(keys).index
            # Set the edge as activation if edge exists in df_g
            df.loc[i1.isin(i2),'type']=1
            # Rename nodes
            df[keys] = df[keys].applymap(lambda x: chr(ord('A') + x))
            # Save the topofile
            df.to_csv('./TOPO/Impure_'+str(n)+'_'+str(m)+'_'+str(g_n)+'.topo',sep='\t',index=False)
            # increment the graph number
            g_n+=1
            if g_n>=nnets:
                break
