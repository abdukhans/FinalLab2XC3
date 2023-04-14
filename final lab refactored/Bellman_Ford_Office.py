import abc
class Bellman_Ford_Office (abc.ABC):
    @abc.abstractmethod
    def calc_sp(self, graph, source, dest):
        pass