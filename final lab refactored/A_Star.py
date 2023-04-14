import min_heap
from A_Star_Office import A_Star_Office
from SPAlgorithm import SPAlgorithm
from HeuristicGraph import HeuristicGraph

class A_Star ( SPAlgorithm ):
    
    __A_Star_Office = None
    
    def __init__(self):
        self.__A_Star_Office = A_Star_Office
    
    def calc_sp(self, graph: HeuristicGraph, source, dest):
        self.__A_Star_Office.calc_sp(graph, source, dest)

s = A_Star()
print( isinstance(s, SPAlgorithm))