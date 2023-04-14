import abc
from HeuristicGraph import HeuristicGraph
class A_Star_Office (abc.ABC):
    @abc.abstractmethod
    def calc_sp(self, graph: HeuristicGraph, source, dest):
        pass