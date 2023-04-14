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
