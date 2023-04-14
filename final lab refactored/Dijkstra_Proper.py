import min_heap
class Dijkstra_Proper ( ):   
    def __init__(self):
        pass
    
    def calc_sp(self, graph, source, dest):
        marked, dist = {}, {}
        Q = min_heap.MinHeap([])
        for i in range(graph.number_of_nodes()):
            marked[i] = False
            dist[i] = float("inf")
            Q.insert(min_heap.Element(i, float("inf")))

        Q.decrease_key(source, 0)
        dist[source] = 0

        while not (Q.is_empty() or marked[dest]):
            current_node = Q.extract_min().value
            marked[current_node] = True
            for neighbour in graph.adj[current_node]:
                edge_weight = graph.w(current_node, neighbour)
                if not marked[neighbour]:
                    if dist[current_node] + edge_weight < dist[neighbour]:
                        dist[neighbour] = dist[current_node] + edge_weight
                        Q.decrease_key(neighbour, dist[neighbour])

        return dist[dest]
    

