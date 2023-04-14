import min_heap

class DirectedWeightedGraph:

    def __init__(self, n):
        self.adj = {}
        self.weights = {}
        for i in range(n):
            self.adj[i] = []

    def are_connected(self, node1, node2):
        for neighbour in self.adj[node1]:
            if neighbour == node2:
                return True
        return False

    def adjacent_nodes(self, node):
        return self.adj[node]

    def add_node(self, node):
        self.adj[node] = []

    def add_edge(self, node1, node2, weight):
        if node2 not in self.adj[node1]:
            self.adj[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def w(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]

    def number_of_nodes(self):
        return len(self.adj)



def dijkstra(G,s,d):

    marked, dist = {}, {}
    Q = min_heap.MinHeap([])
    for i in range(G.number_of_nodes()):
        marked[i] = False
        dist[i] = float("inf")
        Q.insert(min_heap.Element(i, float("inf")))

    Q.decrease_key(s, 0)
    dist[s] = 0

    while not (Q.is_empty() or marked[d]):
        current_node = Q.extract_min().value
        marked[current_node] = True
        for neighbour in G.adj[current_node]:
            edge_weight = G.w(current_node, neighbour)
            if not marked[neighbour]:
                if dist[current_node] + edge_weight < dist[neighbour]:
                    dist[neighbour] = dist[current_node] + edge_weight
                    Q.decrease_key(neighbour, dist[neighbour])

    return dist[d]









g= DirectedWeightedGraph(5)

g.add_edge(0,1,35)
g.add_edge(1,0,35)
g.add_edge(0,2,15)
g.add_edge(2,0,15)
g.add_edge(0,3,25)
g.add_edge(3,0,25)
g.add_edge(1,2,15)
g.add_edge(2,1,15)
g.add_edge(1,4,5)
g.add_edge(4,1,5)
g.add_edge(2,3,35)
g.add_edge(3,2,35)
g.add_edge(2,4,5)
g.add_edge(4,2,5)
g.add_edge(3,4,20)
g.add_edge(4,3,20)

g2 = DirectedWeightedGraph(5)
g2.add_edge(0,1,10)
g2.add_edge(0,2,15)
g2.add_edge(0,3,25)
g2.add_edge(3,4,20)
g2.add_edge(4,1,-40)
g2.add_edge(4,2,-35)
g2.add_edge(3,2,35)

print(dijkstra(g, 0, 3))


"""
def dijkstra(G,s,d):
    marked, dist = {}, {}
    Q = min_heap.MinHeap([])
    for i in range(G.number_of_nodes()):
        marked[i] = False
        dist[i] = float("inf")
        Q.insert(min_heap.Element(i, float("inf")))
    Q.decrease_key(s,0)
    dist[s] = 0

    while not (Q.is_empty() or marked[d]):
        current_node = Q.extract_min().value
        marked[current_node] = True
        for neighbour in G.adj[current_node]:
            edge_weight = G.w[(current_node, neighbour)]
            if edge_weight + dist[current_node] < dist[neighbour]:
                dist[neighbour] = edge_weight + dist[current_node]
                Q.decrease_key(neighbour, edge_weight + dist[current_node])

    return dist[d]
    
    
    def bellman_ford(G, s, d):
    dist = {}
    for i in range(G.number_of_nodes()):
        dist[i] = float("inf")
    dist[s] = 0
    for i in range(G.number_of_nodes()):
        for j in G.adj[i]:
            if dist[j] > dist[i] + G.w(i,j):
                dist[j] = dist[i] + G.w(i,j)
    return dist[d]
"""