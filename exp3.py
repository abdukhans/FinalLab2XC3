import final_project_part1 as fp 
import min_heap
import timeit 
import csv
from tqdm import tqdm
import random as  r
import matplotlib.pyplot as plt

london_map = fp.graph


lines = {}

with open ("london_connections.csv") as file:
    reader = csv.DictReader(file)
    for row in reader:
        st1 = int(row["station1"])
        st2 = int(row["station2"])
        lines[(st1,st2)] = set()

with open  ("london_connections.csv") as file:
    reader = csv.DictReader(file)
    for row in reader:
        st1 = int(row["station1"])
        st2 = int(row["station2"])
        line = int(row["line"])
        lines[(st1,st2)].add(line)
    
    print(reader.line_num)
    

def dijkstra(G, source, dest):
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
        if source == dest:
            break

        for neighbour in G.adj[current_node]:
            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour]:
                Q.decrease_key(neighbour, dist[current_node] + G.w(current_node, neighbour))
                dist[neighbour] = dist[current_node] + G.w(current_node, neighbour)
                pred[neighbour] = current_node
    return dist


def run_time_test(step, connections):
    d_times = []
    a_times = []
    lst_connect = [ i for (id,i) in enumerate(connections) if id%step == 0 ]
    sum_dist = 0 
    for (st1,st2) in tqdm(lst_connect):
            if True:
                start = timeit.default_timer()
                dijkstra(london_map,st2,st1)
                end = timeit.default_timer()
                d_times.append(end - start)            
                h_dict = {}

                for dest in london_map.adj.keys():
                    h_dict[dest] = fp.create_heuristic(fp.stations,st1)(dest)
                
                sum_dist += h_dict[st2]
                
                start =  timeit.default_timer()
                fp.a_star(london_map,st1,st2,h_dict)
                end = timeit.default_timer()

                a_times.append(end- start)

    avg_d_time = sum(d_times)/len(d_times)
    avg_a_time = sum(a_times)/len(a_times)
    avg_dist   = sum_dist/ len(lst_connect)
    imporovement = (avg_d_time/avg_a_time - 1) *100 
    print("Avg dijkstra time     : ", avg_d_time )
    print("Avg A star time       : ", avg_a_time)
    print("A star improvement    : ",  imporovement , "%"  )
    print("Avg distance          : ",  avg_dist  )


def is_intersect(lst1,lst2):
    for i in lst1:
        if i in lst2:
            return True
    return False



def is_same_line (st1,st2,stat_line):
    return is_intersect(stat_line[st1],stat_line[st2]) and st1 != st2



def run_a_star(st1,st2):
    h_dict = {}
    for dest in london_map.adj.keys():
        h_dict[dest] = fp.create_heuristic(fp.stations,st1)(dest)
    
    return fp.a_star(london_map,st1,st2,h_dict)[1]


def get_num_trans(st1 ,st2,stat_line):
    num_trans = 0

    path = run_a_star(st1,st2)

    cur_station = path[0]
    for station in path:
        if not is_same_line(cur_station,station,stat_line) and cur_station != station:
            num_trans += 1 
            cur_station = station

    return num_trans


def create_all_connects(london_map, lst_all_connections):
    for st1 in london_map.adj.keys():
        for st2 in fp.dijkstra(london_map,st1):
            lst_all_connections.append((st1,st2))

lst_all_connections = []
create_all_connects(london_map, lst_all_connections)




def create_stat_line (london_map, lines, lst_lines):
    for i in london_map.adj.keys():
        lst_lines[i] = set()
    for (st1,st2) in lines:
        line = lines[(st1,st2)]
        for line_ in line:
            lst_lines[st1].add(line_)
            lst_lines[st2].add(line_)

def create_line_stat (stat_line , line_stat):
    for i in range  (1,14):
        line_stat[i] = set()

    for station in stat_line:
        for line in stat_line[station]:
            line_stat[line].add(station)

def is_line_adj(l1,l2,line_stat):
    return is_intersect (line_stat[l1],line_stat[l2]) and l1 != l2

def is_station_adj( st1,st2,stat_line,line_stat):

    lines_st1 = stat_line[st1]
    lines_st2 = stat_line[st2]

    for line_1 in lines_st1:
        for line_2 in lines_st2:
            if is_line_adj(line_1,line_2, line_stat):
                return True

    return False


def filter_lst(lst, step ):
    return [i for (id,i ) in enumerate(lst) if id%step == 0 ]

def time_all_stats (lst,step1,step2=1):
    lst_all_connects = filter_lst(lst,step1)
    str_ = "ALL STATIONS"
    print ("\n"+ str_.center(50,"-"))
    run_time_test(step2,lst_all_connects)
    print ((50 )*"-"+"\n\n")

    pass
def time_same_lines_stats (lst, step1,stat_line,step2=1 ):
    lst_filterd = filter_lst(lst,step1)
    lst_same_line_connect = [(st1,st2) for (st1,st2) in tqdm(lst_filterd) if is_same_line(st1,st2, stat_line) ] 
    str_ = "SAME LINES"
    print ("\n"+ str_.center(50,"-"))
    run_time_test(step2,lst_same_line_connect)
    print ((50 )*"-"+"\n\n")



def time_adj_lines_stats (lst,step,stat_line,line_stat,step2=1,moreInfo=False):
    lst_filterd = filter_lst(lst,step)
    lst_adj_line_connect = [(st1,st2) for (st1,st2) in tqdm(lst_filterd) if is_station_adj(st1,st2,stat_line,line_stat) ]

    if moreInfo:  
        num_trans_was_one= 0 
        sum_trans = 0 
        for  i in tqdm(lst_adj_line_connect):
            if get_num_trans(i[0],i[1],stat_line) == 1:
                num_trans_was_one += 1
            sum_trans+= get_num_trans(i[0],i[1],stat_line)


        print("Success rate: " , str(100*(num_trans_was_one / len(lst_adj_line_connect)))+"%" )
        print ( "AVG TRANS: " + str(sum_trans/len(lst_adj_line_connect)))

            


    str_ = "ADJACENT LINES"
    print ("\n"+str_.center(50,"-"))
    run_time_test(step2,lst_adj_line_connect)
    print ((50 )*"-"+"\n\n")


def time_mult_trans_stats(lst,step,stat_line,line_stat,step2=1):
    lst_filterd = filter_lst(lst,step)
    lst_mult_trans = [(st1,st2) for (st1,st2) in tqdm(lst_filterd) if not is_station_adj(st1,st2,stat_line,line_stat) and not is_same_line(st1,st2,stat_line) ]
    str_ = "MULT. TRANSFERS"
    print ("\n"+str_.center(50,"-"))
    run_time_test(step2,lst_mult_trans)
    print ((50 )*"-"+"\n\n")


def time_num_trans_stats(lst,step,num_trans,stat_line,step2=1):
    lst_filterd = filter_lst(lst,step)
    lst_num_trans = [(st1,st2) for (st1,st2) in tqdm(lst_filterd) if get_num_trans(st1,st2,stat_line) == num_trans]
    str_ = "NUM TRANS: " + str(num_trans) +" "
    print ("\n"+str_.center(50,"-"))
    run_time_test(step2,lst_num_trans)
    print ((50 )*"-"+"\n\n")

    


def run_intr_test(lst_connection, step1=100):
    lst_x = []
    lst_y = []
    for (st1,st2) in tqdm(lst_connection):
        if st1 != st2:
            dist_st2 = fp.create_heuristic(fp.stations,st1)(st2)
            d_time = 0 
            a_time = 0 

            intricate = dist_st2
            start = timeit.default_timer()
            dijkstra(london_map,st1,st2)
            end = timeit.default_timer()
            d_time+= (end - start)            
            h_dict = {}

            for dest in london_map.adj.keys():
                h_dict[dest] = fp.create_heuristic(fp.stations,st1)(dest)
                    
                    
            start =  timeit.default_timer()
            fp.a_star(london_map,st1,st2,h_dict)
            end = timeit.default_timer()

            a_time += (end - start)



            imporovement =  (d_time/a_time )  
            lst_x.append(intricate)
            lst_y.append(imporovement)

    plt.title("Relative A* improvement vs Distance")
    plt.ylabel("Relative A* improvement")
    plt.xlabel("Distance")
    plt.scatter(lst_x,lst_y)
    plt.show()



stat_line = {}

create_stat_line(london_map, lines, stat_line)

line_stat = {}

create_line_stat(stat_line,line_stat)

r.shuffle(lst_all_connections)


#---------------  Fig 3.1 --------------
time_all_stats(lst_all_connections,1)

#---------------  Fig 3.2 --------------
time_same_lines_stats(lst_all_connections,10,stat_line)

#---------------  Fig 3.3 --------------
time_adj_lines_stats(lst_all_connections,10,stat_line,line_stat,step2=10)

#---------------  Fig 3.4 --------------
time_mult_trans_stats(lst_all_connections,10,stat_line,line_stat)

#---------------  Fig 3.5 --------------
time_num_trans_stats(lst_all_connections,100,0,stat_line)

#---------------  Fig 3.6 --------------
time_num_trans_stats(lst_all_connections,100,1,stat_line)
time_num_trans_stats(lst_all_connections,100,2,stat_line)
time_num_trans_stats(lst_all_connections,100,3,stat_line)
time_num_trans_stats(lst_all_connections,100,4,stat_line)
time_num_trans_stats(lst_all_connections,10,5,stat_line)

#---------------  Fig 3.7 --------------
run_intr_test(filter_lst(lst_all_connections,100))