import abc
class Graph (abc.ABC):
    @abc.abstractmethod
    def get_adj_nodes(self, node):
        pass

    @abc.abstractmethod
    def add_node(self, node):
        pass

    @abc.abstractmethod
    def add_edge(self, node1, node2, weight):
        pass
    
    @abc.abstractmethod
    def get_num_nodes(self):
        pass

    @abc.abstractmethod
    def w(self, node):
        pass

