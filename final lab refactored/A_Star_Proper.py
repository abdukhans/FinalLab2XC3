import min_heap

class A_Star_Proper ( ):
        
    def __init__(self):
        pass
    
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