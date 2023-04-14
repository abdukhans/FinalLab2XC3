from Dijkstra_Office import Dijkstra_Office
from Dijkstra_Proper import Dijkstra_Proper

class Dijkstra_Adapter ( Dijkstra_Office ):
    
    __Dijkstra_Alg = None
    
    def __init__(self):
        self.__Dijkstra_Alg = Dijkstra_Proper()
    
    def calc_sp(self, graph, source, dest):
        return sum(self.__Dijkstra_Alg(graph, source, dest))

