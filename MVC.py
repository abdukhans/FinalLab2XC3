from collections import deque
import random
import matplotlib.pyplot as plot
import numpy as np
import graph as g 

#Undirected graph using an adjacency list
class Graph:

    def __init__(self, n):
        self.adj = {}
        for i in range(n):
            self.adj[i] = []

    def are_connected(self, node1, node2):
        return node2 in self.adj[node1]

    def adjacent_nodes(self, node):
        return self.adj[node]

    def add_node(self):
        self.adj[len(self.adj)] = []

    def add_edge(self, node1, node2):
        if node1 not in self.adj[node2]:
            self.adj[node1].append(node2)
            self.adj[node2].append(node1)

    def number_of_nodes(self):
        return len(self.adj)
        
def add_to_each(sets, element):
    copy = sets.copy()
    for set in copy:
        set.append(element)
    return copy

def power_set(set):
    if set == []:
        return [[]]
    return power_set(set[1:]) + add_to_each(power_set(set[1:]), set[0])

def is_vertex_cover(G, C):
    for start in G.adj:
        for end in G.adj[start]:
            if not(start in C or end in C):
                return False
    return True

def MVC(G):
    nodes = [i for i in range(len(G.adj))]
    subsets = power_set(nodes)
    min_cover = nodes
    for subset in subsets:
        if is_vertex_cover(G, subset):
            if len(subset) < len(min_cover):
                min_cover = subset
    return min_cover

def has_cycle(G):
    def DFSCycleChecker(node1, parent, marked):
        marked[node1] = True
        for node in G.adjacent_nodes(node1):
            if marked[node] == False:
                if DFSCycleChecker(node, node1, marked) == True:
                    return True
            elif node != parent:
                return True
        return False

    marked = [False] * len(G.adj)
    for i in range(len(G.adj)):
        if marked[i] == False:
            if DFSCycleChecker(i, -1, marked) == True:
                return True
    return False

def is_connected(G):
    def dfs(node, marked):
        marked.add(node)
        for adj in G.adjacent_nodes(node):
            if adj not in marked:
                dfs(adj, marked)

    marked = set()
    dfs(list(G.adj.keys())[0], marked)
    return len(marked) == len(G.adj)


def approx1(G):
    C = set()
    copy = Graph(len(G.adj))
    for node in G.adj:
        for adjacent_node in G.adj[node]:
            copy.add_edge(node, adjacent_node)
    vertex_cover=False
    
    while vertex_cover == False:
        max_degree_node = None
        max_degree = -1
        for node in copy.adj:
            degree = len(copy.adjacent_nodes(node))
            if degree > max_degree:
                max_degree_node = node
                max_degree = degree
        v = max_degree_node
        C.add(v)
        for node in copy.adjacent_nodes(v):
            if (v in copy.adj[node]):
                copy.adj[node].remove(v)
        del copy.adj[v]
        if (is_vertex_cover(G, C)==True):
            vertex_cover=True
    return C

def approx2(G):
    C = set()
    while not is_vertex_cover(G, C):
        v = random.choice([node for node in G.adj if node not in C])
        C.add(v)
    return C
    
def approx3(G):
    
    # 1.
    # copied G so that we can remove the edges accordingly, done to not disrupt structure of original graph
    # this copied graph represents our approximated vertex cover
    C = set()
    copy = Graph(len(G.adj))
    for node in G.adj:
        for adjacent_node in G.adj[node]:
            copy.add_edge(node, adjacent_node)
    
    # dealing with graphs whoses nodes are completely disjoint
    listValues = list(G.adj.values())
    allEmpty = True
    for value in listValues:
        if len(value) > 0:
            allEmpty = False
            break
    
    if allEmpty:
        numNodes = G.number_of_nodes()        
        for node in range(numNodes):
            C.add(node)
        return C
        
    vertex_cover=False
    
    while vertex_cover==False:
        select = False
        
        #2. 
        #randomly selecting u and c vertex to add to copied Graph (which represents our approximated vertex cover)
        while select == False:
            v = random.choice([node for node in G.adj])
            if (len(copy.adj[v])>=1):
                c = random.choice(copy.adj[v])
                select = True
                
        # 3.
        # adding the ranomly choosen edges u and v to the copied graph
        C.add(v)
        C.add(c)
        
        #4. 
        # removing all incident edges from u and v
        for node in copy.adjacent_nodes(v):
            if (v in copy.adj[node]):
                copy.adj[node].remove(v)
            if (c in copy.adj[node]):
                copy.adj[node].remove(c)
        
        #5. 
        # going through steps 2-4 again if we did not already generate the approximated minimum vertex cover
        if (is_vertex_cover(G, C)==True):
            vertex_cover=True
    return C

def size_mcv (mcv):
    return len(mcv)
    
def create_random_graph(i, j):
    G = Graph(i)
    for _ in range(j):
        rand1 = random.randint(0,i - 1)
        rand2 = random.randint(0,i - 1)    
        while ( rand2 in G.adj[rand1] ):
            print (rand1,rand2, "U may have added too many nodes")
            rand1 = random.randint(0,i - 1)
            rand2 = random.randint(0,i - 1) 
        
        G.add_edge (rand1, rand2)
    return G
    
def create_rand_graph_safe (num_nodes, num_edges):
    lst_tups = []
    G = Graph(num_nodes)
    for x in range (num_nodes):
        for  y in range (x , num_nodes):
            lst_tups.append((x,y))
    for i  in range (num_edges):
        if len(lst_tups) == 0 :
            print ("You added to many edges.")
        tup = lst_tups.pop(random.randint(0, len(lst_tups) - 1))
        x = tup[0]
        y = tup[1]
        G.add_edge(x,y)
    return G

#### test cases based on differnet number of edges 
def test1_edge():
    approx1_perf=[]
    for num_edges in range (1,31):
        sum_approx1=0
        sum_mvc=0
        for j in range (1000):
            G1= create_rand_graph_safe(8,num_edges)
            sum_mvc+=size_mcv(MVC(G1))
            sum_approx1 += size_mcv(approx1(G1))
        expected_performance1=sum_approx1/sum_mvc
        approx1_perf.append(expected_performance1)
    return approx1_perf

def test2_edge():
    approx2_perf=[]
    for num_edges in range (1,31):
        sum_approx2=0
        sum_mvc=0
        for j in range (1000):
            G1= create_rand_graph_safe(8,num_edges)
            sum_mvc+=size_mcv(MVC(G1))
            sum_approx2 += size_mcv(approx2(G1))
        expected_performance2=sum_approx2/sum_mvc
        approx2_perf.append(expected_performance2)
    return approx2_perf

def test3_edge():
    approx3_perf=[]
    for num_edges in range (1,31):
        sum_approx3=0
        sum_mvc=0
        for j in range (1000):
            G1= create_rand_graph_safe(8,num_edges)
            sum_mvc+=size_mcv(MVC(G1))
            sum_approx3 += size_mcv(approx3(G1))
        expected_performance3=sum_approx3/sum_mvc
        approx3_perf.append(expected_performance3)
    return approx3_perf

#### graph for test cases based on differnet number of edges (figure Approximation.1)
list_len = len(test1_edge())
x = np.arange(1, list_len + 1)
plot.figure("figure Approximation.1")
plot.plot(x,test1_edge(), label='approx1')
plot.plot(x,test2_edge(), label='approx2')
plot.plot(x,test3_edge(), label='approx3')
plot.title("Edge vs Expected Performance")
plot.xlabel("Number of edges")
plot.ylabel("Expected Performance")
plot.legend()
plot.show()

#### test cases based on differnet number of nodes 
def test1_node():
    approx1_perf=[]
    for num_nodes in range (5,11):
        sum_approx1=0
        sum_mvc=0
        for j in range (1000):
            G1= create_rand_graph_safe(num_nodes,14)
            sum_mvc+=size_mcv(MVC(G1))
            sum_approx1 += size_mcv(approx1(G1))
        expected_performance1=sum_approx1/sum_mvc
        approx1_perf.append(expected_performance1)
    return approx1_perf

def test2_node():
    approx2_perf=[]
    for num_nodes in range (5,11):
        sum_approx2=0
        sum_mvc=0
        for j in range (1000):
            G1= create_rand_graph_safe(num_nodes,14)
            sum_mvc+=size_mcv(MVC(G1))
            sum_approx2 += size_mcv(approx2(G1))
        expected_performance2=sum_approx2/sum_mvc
        approx2_perf.append(expected_performance2)
    return approx2_perf

def test3_node():
    approx3_perf=[]
    for num_nodes in range (5,11):
        sum_approx3=0
        sum_mvc=0
        for j in range (1000):
            G1= create_rand_graph_safe(num_nodes,14)
            sum_mvc+=size_mcv(MVC(G1))
            sum_approx3 += size_mcv(approx3(G1))
        expected_performance3=sum_approx3/sum_mvc
        approx3_perf.append(expected_performance3)
    return approx3_perf

#### graph for test cases based on differnet number of nodes (figure Approximation.2)
list_len = len(test1_node())
x = np.arange(5, 11)
plot.figure("figure Approximation.2")
plot.plot(x,test1_node(), label='approx1')
plot.plot(x,test2_node(), label='approx2')
plot.plot(x,test3_node(), label='approx3')
plot.title("Nodes vs Expected Performance")
plot.xlabel("Number of nodes")
plot.ylabel("Expected Performance")
plot.legend()
plot.show() 


def allGraphs(numNodes):
    # list of all potential edge pairs for a graph with numNodes-number of nodes
    c = []
    for i in range(0, numNodes-1):
        for j in range(i+1, numNodes):
            c.append((i,j))
    
    # generating all possible groups of edge pairs for a graph with numNodes-number of nodes
    pc = power_set(c)
    
    # generating all possible graphs with numNodes-number of nodes
    # each graphs corresponds to a set entry in pc
    graphs = []        
    for edges in pc:
        newGraph = Graph(numNodes)
        for edge in edges:
            newGraph.add_edge(edge[0], edge[1])
        graphs.append(newGraph)
    
    return graphs

#### test cases based on differnet number of nodes
def nTest1_node():
    approx1_perf=[]
    for num_nodes in range (1,6):
        sum_approx1=0
        sum_mvc=0
        
        graphs = allGraphs(num_nodes)
        
        for G1 in graphs:
            sum_mvc += size_mcv(MVC(G1))
            sum_approx1 += size_mcv(approx1(G1))
        
        if sum_mvc > 0:
            expected_performance1=sum_approx1/sum_mvc
            approx1_perf.append(expected_performance1)
        else:
            approx1_perf.append(0)
    return approx1_perf

def nTest2_node():
    approx2_perf=[]
    for num_nodes in range (1,6):
        sum_approx2=0
        sum_mvc=0
        
        graphs = allGraphs(num_nodes)
        
        for G1 in graphs:
            sum_mvc += size_mcv(MVC(G1))
            sum_approx2 += size_mcv(approx2(G1))
            
            print(size_mcv(MVC(G1)), size_mcv(approx2(G1)))
            
         
        if sum_mvc > 0:
            expected_performance2=sum_approx2/sum_mvc
            approx2_perf.append(expected_performance2)
        else:
            approx2_perf.append(0)   

    return approx2_perf

def nTest3_node():
    approx3_perf=[]
    for num_nodes in range (1,6):
        sum_approx3=0
        sum_mvc=0
        
        graphs = allGraphs(num_nodes)
        
        #print(1)
        for G1 in graphs:
            #print(G1.adj)
            sum_mvc += size_mcv(MVC(G1))
            #print(G1.adj)
            sum_approx3 += size_mcv(approx3(G1))
            #print(G1.adj)
            
        if sum_mvc > 0:
            expected_performance3=sum_approx3/sum_mvc
            approx3_perf.append(expected_performance3)
        else:
            approx3_perf.append(0)   
            
    return approx3_perf

#### worst-case analysis based on differnet number of nodes (figure Approximation.3)
list_len = len(nTest1_node())
x = np.arange(1, 6)
plot.figure("figure Approximation.3")
plot.plot(x,nTest1_node(), label='approx1')
plot.plot(x,nTest2_node(), label='approx2')
plot.plot(x,nTest3_node(), label='approx3')
plot.title("Nodes vs Expected Performance")
plot.xlabel("Number of nodes")
plot.ylabel("Expected Performance")
plot.legend()
plot.show() 
