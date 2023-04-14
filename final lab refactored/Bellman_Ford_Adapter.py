from Bellman_Ford_Office import Bellman_Ford_Office
from Bellman_Ford_Proper import Bellman_Ford_Proper

class Bellman_Ford_Adapter ( Bellman_Ford_Office ):
    
    __Bellman_Ford_Alg = None
    
    def __init__(self):
        self.__Bellman_Ford_Alg = Bellman_Ford_Proper()
    
    def calc_sp(self, graph, source, dest):
        return sum(self.__Bellman_Ford_Alg(graph, source, dest))