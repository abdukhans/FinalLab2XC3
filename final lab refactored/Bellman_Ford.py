import min_heap
from Bellman_Ford_Office import Bellman_Ford_Office
from SPAlgorithm import SPAlgorithm

class Bellman_Ford ( SPAlgorithm ):
    
    __Bellman_Ford_Office = None
    
    def __init__(self):
        self.__Bellman_Ford_Office = Bellman_Ford_Office
    
    def calc_sp(self, graph, source, dest):
        self.__Bellman_Ford_Office.calc_sp(graph, source, dest)

