import abc
from Graph import Graph

class WeightedGraph ( Graph ):
    
    def __init__(self):
        self.adj = {}
        self.weights = {}
    
    def get_adj_nodes(self, node):
        return self.adj[node]

    def add_node(self, node):
        self.adj[node] = []

    def add_edge(self, node1, node2, weight):
        if node2 not in self.adj[node1]:
            self.adj[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def get_num_nodes(self):
        return len(self.adj)

    def w(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]
            
    def are_connected(self, node1, node2):
        for neighbour in self.adj[node1]:
            if neighbour == node2:
                return True
        return False

s = WeightedGraph()
print( isinstance(s, Graph))


