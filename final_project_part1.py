import min_heap
import random
import csv
from math import radians, cos, sin, sqrt, atan2
import csv
import time

import itertools

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

#------------------------------------Part 3-----------------------------------
# Parse the stations from the CSV file and store them in a dictionary 
def parse_stations(filename):
    stations = {}
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            id = int(row['id'])
            latitude = float(row['latitude'])
            longitude = float(row['longitude'])
            name = row['name']
            stations[id] = {
                'latitude': latitude,
                'longitude': longitude,
                'name': name
            }
    return stations

# Parse the connections between stations from the CSV file
def parse_connections(filename):
    connections = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            station1 = int(row['station1'])
            station2 = int(row['station2'])
            time = int(row['time'])
            connections.append((station1, station2, time))
    return connections

# Build a directed, weighted graph using the stations and connections data.
# Edge weights are determined by the time it takes to travel between stations.
def build_graph(stations, connections):
    graph = DirectedWeightedGraph()
    for station_id in stations:
        graph.add_node(station_id)
    for station1, station2, time in connections:
        graph.add_edge(station1, station2, time)
        graph.add_edge(station2, station1, time)
    return graph

# Calculate the haversine distance between two points on the Earth's surface
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    central_angle = 2 * atan2(sqrt(a), sqrt(1-a))
    distance=R * central_angle
    return distance

# This function takes the stations data and a destination station ID, then returns
# a function that calculates the straight-line distance between the destination station
# and any other station in the network using the haversine formula.
def create_heuristic(stations, destination):
    lat2, lon2 = stations[destination]['latitude'], stations[destination]['longitude']
    def heuristic(station):
        lat1, lon1 = stations[station]['latitude'], stations[station]['longitude']
        return haversine(lat1, lon1, lat2, lon2)
    return heuristic


stations = parse_stations("london_stations.csv")
connections = parse_connections("london_connections.csv")
graph = build_graph(stations, connections)

#a heuristic function that estimates the cost of reaching station 11 from any other station
heuristic = create_heuristic(stations, 11)
#heuristic value of the station 2 to the destination station 11
heuristic_value = heuristic(2)

print(heuristic_value)

