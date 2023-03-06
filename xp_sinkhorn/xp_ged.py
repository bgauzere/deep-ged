import random
import sys
from greycdata.loaders import load_acyclic, load_MAO
import librariesImport
import gedlibpy
from pyged.ged import GED
from pyged.costfunctions import ConstantCostFunction
from pyged.solvers import SolverLSAP, SolverLSAPE, SolverSinkhorn
import pickle
from tqdm import tqdm
import networkx as nx


def run_xp(graphs, solvers):
    """Parameters
    ----------
    graphs : List of networkx Graphs
    solvers : list[Solver]
        List of solvers to use to compute ged

    Returns
    -----------
    geds : dict[str,list[list[float]]]
        The list of computed geds for each solver in solvers

    """
    def getname(x): return x.__class__.__name__
    cf = ConstantCostFunction(1, 3, 1, 3, label_to_compare="atom_symbol")
    geds = {getname(solver): [] for solver in solvers}
    for g1 in tqdm(graphs):
        for g2 in graphs:
            for solver in solvers:
                ged = GED(cf, solver=solver)
                distance, rho, varrho = ged.ged(g1, g2)
                # print(f"{solver.__class__.__name__} : {distance}")
                geds[getname(solver)].append(distance)

    return geds


def shuffle_nodes(graphs):
    shuffled_graphs = []
    for graph in graphs:
        node_mapping = dict(zip(graph.nodes(), sorted(
            graph.nodes(), key=lambda k: random.random())))
        shuffled_graph = nx.relabel_nodes(graph, node_mapping)
        shuffled_graphs.append(shuffled_graph)
    return shuffled_graphs


def main():
    graphs, properties = load_MAO()
    # shuffle node order
    shuffled_graphs = shuffle_nodes(graphs)

    print("Description du dataset;")
    print(f'Number of graphs: {len(shuffled_graphs)}')
    print(f'Number of features: {shuffled_graphs[0].nodes(data=True)}')
    print("-"*80)
    solvers = [SolverLSAP(), SolverLSAPE(), SolverSinkhorn()]
    geds = run_xp(shuffled_graphs, solvers)
    with open("results_ged", 'wb') as f:
        pickle.dump(geds, f)


if __name__ == '__main__':
    main()
