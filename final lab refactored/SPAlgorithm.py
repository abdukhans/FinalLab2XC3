import abc
class SPAlgorithm (abc.ABC):
    @abc.abstractmethod
    def calc_sp(self, graph, source, dest):
        pass