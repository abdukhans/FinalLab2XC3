from A_Star_Office import A_Star_Office
from A_Star_Proper import A_Star_Proper

class A_Star_Adapter ( A_Star_Office ):
    
    __A_Star_Alg = None
    
    def __init__(self):
        self.__A_Star_Alg = A_Star_Proper()
    
    def calc_sp(self, graph, source, dest):
        return sum(self.__A_Star_Alg(graph, source, dest, graph.get_heuristic())[1])

s = A_Star_Adapter()
print( isinstance(s, A_Star_Office))