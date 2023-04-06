import final_project_part1 as fp
from matplotlib import pyplot as plt
import timeit as t 
from tqdm import tqdm 
import math as m

def run_Algo(G:fp.DirectedWeightedGraph, path_algo, string= "Dijkstra's"):
    n = G.number_of_nodes()
    mat = fp.mystery(G)
    for i in range (n):        
        dist = path_algo(G,i)
        for j in range(n):
            if dist[j]!= mat[i][j] and i!=j:
                string1 = "Error from node i : " + str(i) + " to node j : "+ str(j)
                string2 = string.ljust(len(string1)-3," ")+": "+ str(dist[j])
                string3 = "Mystery".ljust(len(string1)-3," ")+": "+ str(mat[i][j])
                print(string1)
                print(string2)
                print(string3)
                return False 
    return True
    

def run_Dijkstra_test(max_size, max_upper):
    for i in tqdm (range (1,max_size+1)):
        g = fp.create_random_complete_graph(i,max_upper)
        if not(run_Algo(g, fp.dijkstra)):
            return False
    return True

def run_neg_tests(algo,string):
    g_neg = fp.DirectedWeightedGraph() 

    for i in range(4):
        g_neg.add_node(i)

    g_neg.add_edge(0,1, 3)
    g_neg.add_edge(0,2, 5)
    g_neg.add_edge(1,3,-1)
    g_neg.add_edge(2,1,-2)
    g_neg.add_edge(2,3, 5)


    print("\n\nGraph from FigM.2: ", run_Algo(g_neg,algo,string=string))

    g_neg = fp.DirectedWeightedGraph() 

    for i in range(3):
        g_neg.add_node(i)
    g_neg.add_edge(0,1,5)
    g_neg.add_edge(0,2,2)
    g_neg.add_edge(1,2,-4)
    print("Graph from FigM.3: ",run_Algo(g_neg,algo,string=string), "\n")


def run_time_tests(num_runs):

    x = []
    y = []
    for size in tqdm(range (2,num_runs+1)):
        g = fp.create_random_complete_graph(size, size )

        x.append(size)
        start = t.default_timer()
        fp.mystery(g)
        end = t.default_timer()
        y.append(end - start)

    plt.plot(x,y)
    plt.title("Time vs # of vertices")
    plt.xlabel("# of vertices")
    plt.ylabel("Time")
    plt.show()
    return (x,y)

def run_log_test (x,y):
    y2 = [ m.log(time) for time in y  ]
    x2 = [ m.log(size )for size in x  ]
    sum_slope = 0
    for x_d in range(len(x2)  - 1):
        dx = x2[x_d] - x2[x_d + 1]
        dy = y2[x_d] - y2[x_d + 1]
        sum_slope += dy/dx
    print ("--------------------\n\n")
    print(sum_slope/(len(x2) - 1))

    print ("\n\n--------------------")

    plt.plot(x2,y2)
    plt.title("log(Time) vs log(# of vertices)")
    plt.xlabel("log(# of vertices)")
    plt.ylabel("log(Time)")
    plt.show()




max_size = 60


#print ("\n----------\n\n",run_Dijkstra_test(max_size,max_size), "\n\n----------")


num_nodes = 4
max_upper = 2
min_lower = -1

#run_neg_tests(fp.dijkstra,"Dijkstra")

#run_neg_tests(fp.bellman_ford,"BellManFord")


(x,y) = run_time_tests(100)


run_log_test(x,y)











