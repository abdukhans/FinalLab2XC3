from SPAlgorithm import SPAlgorithm

class Bellman_Proper ( SPAlgorithm ):
    def __init__(self):
        pass
    
    def calc_sp(self, graph, source, dest):
        dist = {}
        for i in range(graph.number_of_nodes()):
            dist[i] = float("inf")
        dist[source] = 0
        for i in range(graph.number_of_nodes()):
            for j in graph.adj[i]:
                if dist[j] > dist[i] + graph.w(i,j):
                    dist[j] = dist[i] + graph.w(i,j)
        return dist[dest]
