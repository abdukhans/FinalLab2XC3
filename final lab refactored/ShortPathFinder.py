from Graph import Graph
from WeightedGraph import WeightedGraph
from Dijkstra import Dijkstra


class ShortPathFinder():
    
    __mainGraph = None
    __mainAlgorithm = None
    
    def __init__(self):
        self.__mainGraph = WeightedGraph()
        self.__mainAlgorithm = Dijkstra()
    
    def calc_short_path(self, source, dest):
        return float(self.__mainAlgorithm.calc_sp(self.__mainGraph, source, dest))
        
    def set_graph(self, graph):
        self.__mainGraph = graph
    
    def set_algorithm(self, algorithm):
        self.__mainAlgorithm = algorithm
    
        