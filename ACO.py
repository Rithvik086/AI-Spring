import random
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation

# ------------------------
# Helper function stubs
# ------------------------

def roulette_wheel_selection(candidates, current_node, pheromones, heuristic):
    probabilities = []
    total = 0.0
    for j in candidates:
        tau = pheromones.get((current_node, j), 1.0)
        eta = heuristic.get((current_node, j), 1.0)
        prob = tau * eta
        probabilities.append((j, prob))
        total += prob

    if total == 0:
        return random.choice(candidates)

    r = random.uniform(0, total)
    cumulative = 0
    for j, prob in probabilities:
        cumulative += prob
        if r <= cumulative:
            return j
    return probabilities[-1][0]  # fallback


#whatever this is , is not implemented yet
def two_opt(route,distance_matrix):
    def calulate_total_distance(route):
        distance=distance_matrix["0"][route[0]]
        for i in range(len(route)-1):
            distance+=distance_matrix[route[i]][route[i+1]]
        distance+=distance_matrix[route[-1]]["0"]
        return distance
    
    best_route = route[:]
    best_distance = calulate_total_distance(route)
    improved = True

    while improved:
        improved = False
        for i in range(1,len(best_route)-1):
            for j in range(i+1,len(best_route)):
                if j-1 ==1:
                    continue # Skip adjacent nodes as already connected
                new_route = best_route[:i]  
                new_route[i:j]= best_route[j-1:i-1:-1] # Reverse the segment
                new_distance = calulate_total_distance(new_route)
                if new_distance<best_distance:
                    best_route=new_route
                    best_distance=new_distance
                    improved=True

            route = best_route
    return best_route 
    # # Placeholder for now
    # return route

#charging scheme is not implemented yet
def extract_charging_scheme(route):
    return {"charging_points": [node for node in route if node.startswith("F")]}

# ------------------------
# Real Split function (Prins-style)
# ------------------------

def split_route_prins(tour, demand, capacity, distance_matrix):
    n = len(tour)
    cost = [float('inf')] * (n + 1)
    pred = [-1] * (n + 1)
    cost[0] = 0

    cum_demand = [0] * (n + 1)
    for i in range(n):
        cum_demand[i + 1] = cum_demand[i] + demand.get(tour[i], 0)

    for i in range(n):
        for j in range(i + 1, n + 1):
            load = cum_demand[j] - cum_demand[i]
            if load > capacity:
                break

            segment = tour[i:j]
            seg_cost = distance_matrix["0"][segment[0]]
            for k in range(len(segment) - 1):
                seg_cost += distance_matrix[segment[k]][segment[k + 1]]
            seg_cost += distance_matrix[segment[-1]]["0"]

            if cost[i] + seg_cost < cost[j]:
                cost[j] = cost[i] + seg_cost
                pred[j] = i

    segments = []
    idx = n
    while idx > 0:
        i = pred[idx]
        segments.insert(0, tour[i:idx])
        idx = i

    return segments

# ------------------------
# Cost function
# ------------------------

def route_cost(route_segments):
    return sum(len(segment) for segment in route_segments)

# ------------------------
# Main ACO function
# ------------------------

def ant_colony_optimization(I, F, N, pheromones, heuristic, demand, capacity, distance_matrix):
    #I is the set of customers
    #F is the set of charging stations
    #N is population size
    I_hat = set(["0"]) | set(I) | set(F)
    P = []
    k = 1

    while k <= N:
        r = []
        available_nodes = list(I_hat - {"0"})

        i = random.choice(available_nodes)
        r.append(i)
        available_nodes.remove(i)

        while available_nodes:
            current_node = r[-1]
            # Select next node using roulette wheel selection
            j = roulette_wheel_selection(available_nodes, current_node, pheromones, heuristic)
            r.append(j)
            available_nodes.remove(j)
            #we will pe using prins algorithm to split a large trip into smaller (spliting a large route to multiple feasible subroutes)
        x = split_route_prins(r, demand, capacity, distance_matrix)
        x = [two_opt(route,distance_matrix) for route in x]
        P.append(x)
        k += 1

    P_best = min(P, key=route_cost)
    P_best = [two_opt(route,distance_matrix) for route in P_best]
    ls = extract_charging_scheme([n for seg in P_best for n in seg])
#Pbest is the best solution found in this iteration
    return P_best, ls

# ------------------------
# Example Usage
# ------------------------

def visualize_solution(best_solution, distance_matrix, charging_scheme):
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()

    # Collect all nodes and edges
    all_nodes = set(["0"])  # include depot
    edges = []

    for segment in best_solution:
        route = ["0"] + segment + ["0"]
        all_nodes.update(route)
        for i in range(len(route) - 1):
            edges.append((route[i], route[i + 1]))

    G.add_nodes_from(all_nodes)
    G.add_edges_from(edges)

    # Assign positions AFTER nodes are in G
    pos = nx.spring_layout(G, seed=42)

    # Color nodes based on type
    node_colors = []
    for node in G.nodes():
        if node == "0":
            node_colors.append("green")  # Depot
        elif node.startswith("F"):
            node_colors.append("blue")   # Charging station
        else:
            node_colors.append("orange") # Customer

    # Draw everything
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, edgecolors='black')
    nx.draw_networkx_labels(G, pos)

    edge_labels = {(u, v): distance_matrix[u][v] for u, v in G.edges if v in distance_matrix[u]}
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Highlight used charging stations (optional)
    if "charging_points" in charging_scheme:
        for node in charging_scheme["charging_points"]:
            if node in G.nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color="cyan", node_size=700, edgecolors='black')

    plt.title("Ant Colony Optimization Route with Charging Stations")
    plt.axis('off')
    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    I = ["1", "2", "3","4","5","6","7","8","9","10"] 
    F = ["F1","F2","F3"]
    N = 5
    capacity = 10
    demand = {"1": 2, "2": 3, "3": 4,"4":5,"5":6,"6":7,"7":8,"8":9,"9":9,"10":1,  "F1": 0, "F2": 0,"F3": 0}

    nodes = ["0"] + I + F
    distance_matrix = {i: {} for i in nodes}
    for i in nodes:
        for j in nodes:
            if i != j:
                distance_matrix[i][j] = abs(int(i.strip("F")) - int(j.strip("F"))) + 1 if i.strip("F").isdigit() and j.strip("F").isdigit() else 5
    # the phermone matrix is set with values 1 for now as no chargin scheme is implemented
    pheromones = {(i, j): 1.0 for i in nodes for j in nodes if i != j}
    heuristic = {(i, j): 1.0 / distance_matrix[i][j] for i in nodes for j in nodes if i != j}

    best_solution, charging_scheme = ant_colony_optimization(I, F, N, pheromones, heuristic, demand, capacity, distance_matrix)
    print("Best Solution:", best_solution)
    print("Charging Scheme:", charging_scheme)
    visualize_solution(best_solution, distance_matrix, charging_scheme)
    # Animate one of the subroutes (for demo)



