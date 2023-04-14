import min_heap
import random
import matplotlib.pyplot as plot
import numpy as np

class DirectedWeightedGraph:

    def __init__(self):
        self.adj = {}
        self.weights = {}

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


def dijkstra(G, source):
    pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {} #Distance dictionary
    Q = min_heap.MinHeap([])
    nodes = list(G.adj.keys())

    #Initialize priority queue/heap and distances
    for node in nodes:
        Q.insert(min_heap.Element(node, float("inf")))
        dist[node] = float("inf")
    Q.decrease_key(source, 0)

    #Meat of the algorithm
    while not Q.is_empty():
        current_element = Q.extract_min()
        current_node = current_element.value
        dist[current_node] = current_element.key
        for neighbour in G.adj[current_node]:
            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour]:
                Q.decrease_key(neighbour, dist[current_node] + G.w(current_node, neighbour))
                dist[neighbour] = dist[current_node] + G.w(current_node, neighbour)
                pred[neighbour] = current_node
    return dist

def dijkstra_approx(G, source, k):
    pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {} #Distance dictionary
    Q = min_heap.MinHeap([])
    nodes = list(G.adj.keys())
    relax={}
    
    #Initialize priority queue/heap and distances
    for node in nodes:
        Q.insert(min_heap.Element(node, float("inf")))
        dist[node] = float("inf")
    Q.decrease_key(source, 0)
    
        #Initialize relax times to 0
    for node in nodes:
        relax[node] = 0
        
    #Meat of the algorithm
    while not Q.is_empty():
        current_element = Q.extract_min()
        current_node = current_element.value
        dist[current_node] = current_element.key
        for neighbour in G.adj[current_node]:
            #relax each node at most k times
            if (dist[current_node] + G.w(current_node, neighbour) < dist[neighbour]) and (relax[neighbour]<k):
                Q.decrease_key(neighbour, dist[current_node] + G.w(current_node, neighbour))
                dist[neighbour] = dist[current_node] + G.w(current_node, neighbour)
                pred[neighbour] = current_node
                relax[neighbour]+=1
    return dist

def bellman_ford(G, source):
    pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {} #Distance dictionary
    nodes = list(G.adj.keys())

    #Initialize distances
    for node in nodes:
        dist[node] = float("inf")
    dist[source] = 0

    #Meat of the algorithm
    for _ in range(G.number_of_nodes()):
        for node in nodes:
            for neighbour in G.adj[node]:
                if dist[neighbour] > dist[node] + G.w(node, neighbour):
                    dist[neighbour] = dist[node] + G.w(node, neighbour)
                    pred[neighbour] = node
    return dist


def bellman_ford_approx(G, source, k):
    pred = {} #Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {} #Distance dictionary
    nodes = list(G.adj.keys())
    relax={}
    
    #Initialize distances
    for node in nodes:
        dist[node] = float("inf")
        #Initialize relax times to 0
        relax[node] = 0
    dist[source] = 0

    #Meat of the algorithm
    for _ in range(G.number_of_nodes()):
        for node in nodes:
            for neighbour in G.adj[node]:
                #relax each node at most k times
                if (dist[neighbour] > dist[node] + G.w(node, neighbour)) and (relax[neighbour]<k):
                    relax[neighbour]+=1
                    dist[neighbour] = dist[node] + G.w(node, neighbour)
                    pred[neighbour] = node
    return dist
    
def total_dist(dist):
    total = 0
    for key in dist.keys():
        total += dist[key]
    return total

def create_random_complete_graph(n,upper):
    G = DirectedWeightedGraph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(n):
            if i != j:
                G.add_edge(i,j,random.randint(1,upper))
    return G


#Assumes G represents its nodes as integers 0,1,...,(n-1)
def mystery(G):
    n = G.number_of_nodes()
    d = init_d(G)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][j] > d[i][k] + d[k][j]: 
                    d[i][j] = d[i][k] + d[k][j]
    return d

def init_d(G):
    n = G.number_of_nodes()
    d = [[float("inf") for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if G.are_connected(i, j):
                d[i][j] = G.w(i, j)
        d[i][i] = 0
    return d

#---------------------------part 2----------------------------
def a_star(G, s, d, h):
    Q = min_heap.MinHeap()
    Q.insert(s, h(s))
    pred = {}
    dist = {s: 0}

    while not Q.is_empty():
        u = Q.extract_min()
        if u == d:
            return pred, dist[d]

        for v in G.adjacent_nodes(u):
            w = G.w(u, v)
            if w is None:
                continue
            if v not in dist or dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u
                Q.insert(v, dist[v] + h(v))

    return {}, None
    

def a_star(G, s, d, h):
    pred = {}
    dist = {}
    Q = min_heap.MinHeap([])
    nodes = list(G.adj.keys())

    # Initialize priority queue/heap and distances
    for node in nodes:
        Q.insert(min_heap.Element(node, float("inf")))
        dist[node] = float("inf")
    Q.decrease_key(s, 0)

    # Meat of the algorithm
    while not Q.is_empty():
        current_element = Q.extract_min()
        current_node = current_element.value
        dist[current_node] = current_element.key

        if current_node == d:
            break

        for neighbour in G.adj[current_node]:
            new_cost = dist[current_node] + G.w(current_node, neighbour)
            if new_cost < dist[neighbour]:
                Q.decrease_key(neighbour, new_cost + h[neighbour])
                dist[neighbour] = new_cost
                pred[neighbour] = current_node

    # path
    path = []
    node = d
    while node in pred:
        path.append(node)
        node = pred[node]
    path.append(s)
    path.reverse()

    return pred, path
#-------------------------------Testing-------------------------------

#---------------------------dijkstra_approx testing-------------------
# Test case 1: Simple graph
G1 = DirectedWeightedGraph()
for node in range(4):
    G1.add_node(node)
G1.add_edge(0, 1, 1)
G1.add_edge(0, 2, 4)
G1.add_edge(1, 2, 2)
G1.add_edge(1, 3, 5)
G1.add_edge(2, 3, 1)

# Expected output
expected_output_1 = {
    0: 0,
    1: 1,
    2: 3,
    3: 4
}
assert dijkstra_approx(G1, 0, 2) == expected_output_1

# Test case 2: Larger graph with varied weights
G2 = DirectedWeightedGraph()
for node in range(5):
    G2.add_node(node)
G2.add_edge(0, 1, 3)
G2.add_edge(0, 3, 7)
G2.add_edge(1, 2, 1)
G2.add_edge(1, 3, 4)
G2.add_edge(2, 4, 2)
G2.add_edge(3, 2, 2)
G2.add_edge(3, 4, 5)
G2.add_edge(4, 1, 1)

# Expected output
expected_output_2 = {
    0: 0,
    1: 3,
    2: 4,
    3: 7,
    4: 6
}
assert dijkstra_approx(G2, 0, 3) == expected_output_2

# Test case 3: Disconnected graph
G3 = DirectedWeightedGraph()
for node in range(6):
    G3.add_node(node)
G3.add_edge(0, 1, 1)
G3.add_edge(1, 2, 2)
G3.add_edge(3, 4, 3)
G3.add_edge(4, 5, 4)

# Expected output
expected_output_3 = {
    0: 0,
    1: 1,
    2: 3,
    3: float("inf"),
    4: float("inf"),
    5: float("inf")
}
assert dijkstra_approx(G3, 0, 2) == expected_output_3

#---------------------------bellman_ford_approx testing-------------------
# Test case 1: Simple graph
G1 = DirectedWeightedGraph()
for node in range(4):
    G1.add_node(node)
G1.add_edge(0, 1, 1)
G1.add_edge(0, 2, 4)
G1.add_edge(1, 2, 2)
G1.add_edge(1, 3, 5)
G1.add_edge(2, 3, 1)

# Expected output
expected_output_1 = {
    0: 0,
    1: 1,
    2: 3,
    3: 4
}
assert bellman_ford_approx(G1, 0, 2) == expected_output_1


# Test case 3: Larger graph with varied weights
G3 = DirectedWeightedGraph()
for node in range(5):
    G3.add_node(node)
G3.add_edge(0, 1, 3)
G3.add_edge(0, 3, 7)
G3.add_edge(1, 2, 1)
G3.add_edge(1, 3, 4)
G3.add_edge(2, 4, 2)
G3.add_edge(3, 2, 2)
G3.add_edge(3, 4, 5)
G3.add_edge(4, 1, 1)

# Expected output
expected_output_3 = {
    0: 0,
    1: 3,
    2: 4,
    3: 7,
    4: 6
}
assert bellman_ford_approx(G3, 0, 3) == expected_output_3


#----------------------------Part 2 test cases---------------------------
# Test case 1: A simple graph with a straight line from s to d
G = DirectedWeightedGraph()
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_edge(1, 2, 2)
G.add_edge(2, 3, 2)
G.add_edge(1, 3, 4)
h = {1: 0, 2: 2, 3: 4}
pred, path = a_star(G, 1, 3, h)
assert path == [1, 3]

# Test case 2: A simple graph with an indirect path from s to d
G = DirectedWeightedGraph()
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_edge(1, 2, 1)
G.add_edge(2, 3, 1)
G.add_edge(1, 3, 5)
h = {1: 0, 2: 2, 3: 1}
pred, path = a_star(G, 1, 3, h)
assert path == [1, 2, 3]

# Test case 3: A graph with multiple paths from s to d
G = DirectedWeightedGraph()
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node(4)
G.add_edge(1, 2, 3)
G.add_edge(1, 3, 2)
G.add_edge(2, 4, 1)
G.add_edge(3, 4, 3)
h = {1: 0, 2: 1, 3: 2, 4: 1}
pred, path = a_star(G, 1, 4, h)
assert path == [1, 2, 4]

# Test case 4: A graph with negative edge weights
G = DirectedWeightedGraph()
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_edge(1, 2, 2)
G.add_edge(2, 3, -1)
G.add_edge(1, 3, 1)
h = {1: 0, 2: 2, 3: 1}
pred, path = a_star(G, 1, 3, h)
assert path == [1, 3]

# Test case 5: A disconnected graph
G = DirectedWeightedGraph()
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_edge(1, 2, 2)
h = {1: 0, 2: 2, 3: 1}
pred, path = a_star(G, 1, 3, h)
assert path == [1]






# Figure 1.1
def dsize_test():
    size_perf=[]
    k = 3
    source = 0
    for num_nodes in range (1,100):
        
        graph = create_random_complete_graph(num_nodes, 60)
        dGraph = dijkstra_approx(graph, source, k)
        

        size_perf.append(total_dist(dGraph))   
            
    return size_perf

def bsize_test():
    size_perf=[]
    k = 3
    source = 0
    for num_nodes in range (1,100):
        
        graph = create_random_complete_graph(num_nodes, 60)
        bGraph = bellman_ford_approx(graph, source, k)

        size_perf.append(total_dist(bGraph))   
            
    return size_perf



list_len = len(dsize_test())
x = np.arange(1, 100)
plot.figure("figure Approximation 1")
plot.plot(x,dsize_test(), label='dijkstra')
plot.plot(x,bsize_test(), label='bellman_ford')
plot.title("Number of Nodes vs Total Distance of Graph")
plot.xlabel("Number of nodes")
plot.ylabel("Total Distance of Graph")
plot.legend()
plot.show() 

# Figure 1.2
def dDensity_test():
    size_perf=[]
    source = 0
    k = 3
    for density in range (1,100):
        
        graph = create_random_complete_graph(60, density)
        dGraph = dijkstra_approx(graph, source, k)

        size_perf.append(total_dist(dGraph))   
            
    return size_perf

def bDensity_test():
    size_perf=[]
    k = 3
    source = 0
    for density in range (1,100):
        
        graph = create_random_complete_graph(60, density)
        bGraph = bellman_ford_approx(graph, source, k)

        size_perf.append(total_dist(bGraph))   
            
    return size_perf


list_len = len(dDensity_test())
x = np.arange(1, 100)
plot.figure("figure Approximation 2")
plot.plot(x,dDensity_test(), label='dijkstra')
plot.plot(x,bDensity_test(), label='bellman_ford')
plot.title("Density of Graph vs Total Distance of Graph")
plot.xlabel("Density of Graphs")
plot.ylabel("Total Distance of Graph")
plot.legend()
plot.show()   

# Figure 1.3
def dK_test():
    size_perf=[]
    source = 0
    for k in range (1,100):
        
        graph = create_random_complete_graph(60, 60)
        dGraph = dijkstra_approx(graph, source, k)

        size_perf.append(total_dist(dGraph))   
            
    return size_perf

def bK_test():
    size_perf=[]
    
    source = 0
    for k in range (1,100):
        
        graph = create_random_complete_graph(60, 60)
        dGraph = bellman_ford_approx(graph, source, k)

        size_perf.append(total_dist(dGraph))   
            
    return size_perf


list_len = len(dK_test())
x = np.arange(1, 100)
plot.figure("figure Approximation 3")
plot.plot(x,dK_test(), label='dijkstra')
plot.plot(x,bK_test(), label='bellman_ford')
plot.title("K vs Total Distance of Graph")
plot.xlabel("K")
plot.ylabel("Total Distance of Graph")
plot.legend()
plot.show()







# Figure 1.4
def dAvsD1():
    size_perf=[]
    k = 3
    source = 0
    for num_nodes in range (2,100):
        
        graph = create_random_complete_graph(num_nodes, 60)
        dGraph = dijkstra_approx(graph, source, k)
        daGraph = dijkstra(graph, source)

        size_perf.append(total_dist(daGraph)/total_dist(dGraph))   
            
    return size_perf


def bAvsB1():
    size_perf=[]
    k = 3
    source = 0
    for num_nodes in range (2,100):
        
        graph = create_random_complete_graph(num_nodes, 60)
        bGraph = bellman_ford_approx(graph, source, k)
        baGraph = bellman_ford(graph, source)

        size_perf.append(total_dist(baGraph)/total_dist(bGraph))   
            
    return size_perf


list_len = len(bAvsB1())
x = np.arange(2, 100)
plot.figure("figure Approximation 4")
plot.plot(x,dAvsD1(), label='dijkstra')
plot.plot(x,bAvsB1(), label='bellman-ford')
plot.title("Number of Nodes vs Ratio Distance of Approximated Graphs relative to Normal")
plot.xlabel("Number of nodes")
plot.ylabel("Ratio Distance of Approximated Graphs relative to Normal Graphs")
plot.legend()
plot.show() 

# Figure 1.5
def dAvsD2():
    size_perf=[]
    k = 3
    source = 0
    for density in range (1,100):
        
        graph = create_random_complete_graph(60, density)
        dGraph = dijkstra_approx(graph, source, k)
        daGraph = dijkstra(graph, source)

        size_perf.append(total_dist(daGraph)/total_dist(dGraph))   
            
    return size_perf


def bAvsB2():
    size_perf=[]
    k = 3
    source = 0
    for density in range (1,100):
        
        graph = create_random_complete_graph(60, density)
        bGraph = bellman_ford_approx(graph, source, k)
        baGraph = bellman_ford(graph, source)

        size_perf.append(total_dist(baGraph)/total_dist(bGraph))   
            
    return size_perf



list_len = len(bAvsB2())
x = np.arange(1, 100)
plot.figure("figure Approximation 5")
plot.plot(x,dAvsD2(), label='dijkstra')
plot.plot(x,bAvsB2(), label='bellman-ford')
plot.title("Density of Graphs vs Ratio Distance of Approximated Graphs relative to Normal")
plot.xlabel("Density of Graphs")
plot.ylabel("Ratio Distance of Approximated Graphs relative to Normal Graphs")
plot.legend()
plot.show() 

# Figure 1.6
def dAvsD3():
    size_perf=[]
    source = 0
    for k in range (1,100):
        
        graph = create_random_complete_graph(60, 60)
        dGraph = dijkstra_approx(graph, source, k)
        daGraph = dijkstra(graph, source)

        size_perf.append(total_dist(daGraph)/total_dist(dGraph))   
            
    return size_perf


def bAvsB3():
    size_perf=[]
    source = 0
    for k in range (1,100):
        
        graph = create_random_complete_graph(60, 60)
        bGraph = bellman_ford_approx(graph, source, k)
        baGraph = bellman_ford(graph, source)

        size_perf.append(total_dist(baGraph)/total_dist(bGraph))   
            
    return size_perf


list_len = len(bAvsB3())
x = np.arange(1, 100)
plot.figure("figure Approximation 6")
plot.plot(x,dAvsD3(), label='dijkstra')
plot.plot(x,bAvsB3(), label='bellman-ford')
plot.title("Value of K vs Ratio Distance of Approximated Graphs relative to Normal")
plot.xlabel("Value of K")
plot.ylabel("Ratio Distance of Approximated Graphs relative to Normal Graphs")
plot.legend()
plot.show() 