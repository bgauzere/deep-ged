from greycdata.loaders import load_acyclic, load_MAO
import librariesImport
import gedlibpy
from pyged.ged import GED
from pyged.costfunctions import ConstantCostFunction
from pyged.solvers import SolverLSAP, SolverLSAPE, SolverSinkhorn


def main():

    graphs, properties = load_MAO()

    print(f'Number of graphs: {len(graphs)}')
    print(f'Number of features: {graphs[0].nodes(data=True)}')
    cf = ConstantCostFunction(1, 3, 1, 3)
    solvers = [SolverLSAP(), SolverLSAPE(), SolverSinkhorn()]
    for solver in solvers:
        print(solver)
        ged = GED(cf, solver=solver)
        distance, rho, varrho = ged.ged(graphs[44], graphs[1])
        print(distance)


if __name__ == '__main__':
    main()
