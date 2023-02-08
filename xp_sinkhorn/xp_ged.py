from greycdata.loaders import load_acyclic, load_MAO
import librariesImport
import gedlibpy
from pyged.ged import GED
from pyged.costfunctions import ConstantCostFunction


def main():

    graphs, properties = load_MAO()

    print(f'Number of graphs: {len(graphs)}')
    print(f'Number of features: {graphs[0].nodes(data=True)}')
    cf = ConstantCostFunction(1, 3, 1, 3)
    ged = GED(cf)
    distance, rho, varrho = ged.ged(graphs[0], graphs[1])
    print(distance)


if __name__ == '__main__':
    main()
