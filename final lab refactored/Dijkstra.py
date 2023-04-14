import min_heap
from SPAlgorithm import SPAlgorithm
from Dijkstra_Office import Dijkstra_Office

class Dijkstra ( SPAlgorithm ):

    __Dijkstra_Office = None
    
    def __init__(self):
        self.__Dijkstra_Office = Dijkstra_Office
    
    def calc_sp(self, graph, source, dest):
        self.__Dijkstra_Office.calc_sp(graph, source, dest)
    
    
