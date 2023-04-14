from WeightedGraph import WeightedGraph

from typing import TypedDict

class Heuristic (TypedDict):
    val1: int
    val2: float
    
class HeuristicGraph ( WeightedGraph ):
    
    __heuristic: Heuristic = None
    
    def __init__(self, heuristic):
        super().__init__()
        self.__heuristic = heuristic
    
    def get_adj_nodes(self, node):
        super().get_adj_nodes(node)

    def add_node(self, node):
        super().add_node(node)

    def add_edge(self, node1, node2, weight):
        super().add_edge(node1, node2, weight)

    def get_num_nodes(self):
        super().get_num_nodes()

    def w(self, node1, node2):
        super().w(node1, node2)
            
    def are_connected(self, node1, node2):
        super().are_connected(node1, node2)
    
    def get_heuristic(self):
        return self.__heuristic
    
    def set_heuristic(self, graph, dest):
        # no implementation as it was not required in the lab
        pass
