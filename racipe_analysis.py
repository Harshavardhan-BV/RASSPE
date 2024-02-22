import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

def get_param(topo:str, i:int):
    """
    Get the RACIPE parameters from the specified topology for a particular replicate. Automatically adds the headers according to the prs file.

    Parameters:
    topo (str): The name of the topofile.
    i (int): Replicate number.

    Returns:
    pandas.DataFrame: The parameters dataframe.
    """
    # Read the prs file
    df_prs = pd.read_csv('./Results/'+topo+'/'+str(i)+'/'+topo+'.prs', sep='\t').Parameter
    # Append Model_index Number_of_stable_states to df_prs
    df_prs = pd.concat([pd.Series(['Model_index', 'Number_of_stable_states']), df_prs])
    # Read the parameters file
    df_param = pd.read_csv('./Results/'+topo+'/'+str(i)+'/'+topo+'_parameters.dat', sep='\t', header=None, names=df_prs, index_col=0)
    return df_param

def get_sol(topo:str, i:int):
    """
    Get the RACIPE solutions from the specified topology for a particular replicate. Automatically adds the headers according to the cfg file.

    Parameters:
    topo (str): The name of the topofile.
    i (int): Replicate number.

    Returns:
    pandas.DataFrame: The solutions dataframe.
    """
    # Read the cfg file
    df_cfg = pd.read_csv('./Results/'+topo+'/'+str(i)+'/'+topo+'.cfg', sep='\t', header=None, names=['Parameter', 'Value', 'bruh', 'bruhh'], index_col=0)
    # Find index range of Genes
    start = df_cfg.index.get_loc('NumberOfGenes')+1
    end = int(df_cfg.loc['NumberOfGenes','Value'])+start
    # Select only the genes
    df_cfg = df_cfg.iloc[start:end, 0]
    df_cfg = pd.concat([pd.Series(['Model_index','Number_of_stable_states','heh']), df_cfg])
    # Read the solution file
    df_sol = pd.read_csv('./Results/'+topo+'/'+str(i)+'/'+topo+'_solution.dat', sep='\t', header=None, index_col=0, names=df_cfg)
    return df_sol

def get_all_param(topo:str, n_repl:int):
    """
    Get all parameters for a given topology across all replicates.

    Parameters:
    topo (str): The name of the topofile.
    n_repl (int): The number of replicates.

    Returns:
    df_param (DataFrame): A DataFrame containing all the parameters.
    """
    df_param = get_param(topo,1)
    for i in range(2,n_repl+1):
        df_param = pd.concat([df_param, get_param(topo,i)])
    return df_param

def get_all_sol(topo:str, n_repl:int):
    """
    Get all parameters for a given topology across all replicates.

    Parameters:
    topo (str): The name of the topofile.
    n_repl (int): The number of replicates.

    Returns:
    df_param (DataFrame): A DataFrame containing all the parameters.
    """
    df_sol = get_sol(topo,1)
    for i in range(2,n_repl+1):
        df_sol = pd.concat([df_sol, get_sol(topo,i)])
    return df_sol

def plot_states(df_sol, topo:str, save=False):
    """
    Plot the distribution of solutions.

    Parameters:
    df_sol (pd.DataFrame): The solutions dataframe.
    topo (str): The name of the topofile.
    save (bool): Whether to save the plot or not. Default is False.

    Returns:
    None

    Saves:
    ./figures/States/{topo}_states.png (File): The plot of the distribution of states.
    """
    # Select only the states
    df_sol = df_sol.iloc[:,2:]
    df_sol = df_sol.melt(var_name='Gene', value_name='State')
    # Plot the distribution of states
    sns.kdeplot(data=df_sol, x='State', hue='Gene', fill=True)
    plt.title(topo)
    if save:
        os.makedirs('./figures/States/', exist_ok=True)
        plt.savefig('./figures/States/'+topo+'_states.png')
        plt.close()
    else:
        plt.show()

def gkNorm(sol,param):
    """
    Normalize the RACIPE solutions using the g/k normalization.

    Parameters:
    sol (pd.DataFrame): The solutions dataframe.
    param (pd.DataFrame): The parameters dataframe.

    Returns:
    pd.DataFrame: The normalized solutions dataframe.
    """
    # List the nodes in sol
    nodes = sol.iloc[:,2:].columns
    # Get the g's for each node
    g = param['Prod_of_'+nodes]
    g.columns = nodes
    # Get the k's for each node
    k = param['Deg_of_'+nodes]
    k.columns = nodes
    # Do the normalization
    sol.iloc[:,2:] = (2**sol.loc[:,nodes])*k.loc[:,nodes]/g.loc[:,nodes]
    return sol

def TopoToAdj(topo:str, plot:bool=True):
    """
    Convert a topofile to an adjacency matrix.

    Parameters:
    topo (str): The name of the topofile.
    plot (bool): Whether to plot the adjacency matrix or not. Default is True.

    Returns:
    pd.DataFrame: The adjacency matrix.

    Saves:
    ./figures/AdjMat_{topo}.svg (File): The plot of the adjacency matrix.
    """
    # Read the topofile as a dataframe
    df = pd.read_csv('./TOPO/'+topo+'.topo',sep='\t')
    # Replace 2 with -1 for column 2
    df[df.columns[2]] = df[df.columns[2]].replace(2,-1)
    # Generate the graph
    G = nx.from_pandas_edgelist(df, source=df.columns[0], target=df.columns[1], edge_attr=df.columns[2], create_using=nx.DiGraph)
    # Convert to adjacency matrix
    adjMat = nx.to_pandas_adjacency(G, weight=df.columns[2])
    if plot:
        sns.clustermap(adjMat, cmap='coolwarm', vmin=-1, vmax=1)
        plt.savefig(f'./figures/AdjMat_{topo}.svg')
        plt.clf()
        plt.close()
    return adjMat

def TopoToInfl(topo:str, lmax:int=10, plot:bool=True):
    """
    Convert a topofile to an influence matrix.

    Parameters:
    topo (str): The name of the topofile.
    lmax (int): The maximum path length of influence. Default is 10.
    plot (bool): Whether to plot the influence matrix or not. Default is True.

    Returns:
    pd.DataFrame: The influence matrix.

    Saves:
    ./figures/InflMat_{topo}_{lmax}.svg (File): The plot of the influence matrix.
    """
    adjMat = TopoToAdj(topo, plot=False)
    nodes = adjMat.columns
    adjMat = adjMat.to_numpy()
    # Set the element to 1 if non-zero
    adjMax = adjMat.copy()
    adjMax[adjMax != 0] = 1.0
    # Initialise the InfluenceMatrix, numerator and denominator
    InflMat = np.zeros_like(adjMat)
    num = np.identity(adjMat.shape[0])
    den = np.identity(adjMat.shape[0])
    for i in range(1,lmax+1):
        num = np.matmul(num, adjMat)
        den = np.matmul(den, adjMax)
        # If denominator is zero, set to 0 to avoid division by zero
        res = np.divide(num, den, out=np.zeros_like(num), where=den!=0)
        # Change nan to zero
        res = np.nan_to_num(res)
        # Add the result to the InfluenceMatrix
        InflMat += res
    # Normalise the matrix
    InflMat = InflMat/lmax
    # Convert to datafram
    InflMat = pd.DataFrame(InflMat, index=nodes, columns=nodes)
    if plot:
        sns.clustermap(InflMat, cmap='coolwarm', vmin=-1, vmax=1)
        plt.savefig(f'./figures/InflMat_{topo}_{lmax}.svg')
        plt.clf()
        plt.close()
    return InflMat