import random
import matplotlib.pyplot as plt
import networkx as nx

# ------------------------
# Helper Functions
# ------------------------

def roulette_wheel_selection(candidates, current_node, pheromones, heuristic):
   
    if not candidates:
        raise ValueError("No valid candidates available for selection.")
    
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
    return probabilities[-1][0]

def two_opt(route, distance_matrix, depot="0"):
    """
    Applies 2-opt local search to improve a route.
    
    Args:
        route (list): List of nodes in the route.
        distance_matrix (dict): Distance matrix.
        depot (str): Depot node ID.
    
    Returns:
        list: Improved route.
    """
    def calculate_total_distance(route):
        if not route:
            return 0
        distance = distance_matrix[depot][route[0]]
        for i in range(len(route) - 1):
            distance += distance_matrix[route[i]][route[i + 1]]
        distance += distance_matrix[route[-1]][depot]
        return distance
    
    best_route = route[:]
    best_distance = calculate_total_distance(route)
    improved = True

    while improved:
        improved = False
        for i in range(1, len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                if j - i == 1:  # Skip adjacent nodes
                    continue
                new_route = best_route[:i] + best_route[j-1:i-1:-1] + best_route[j:]
                new_distance = calculate_total_distance(new_route)
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improved = True
    return best_route



def genetic_algorithm(population, distance_matrix, generations=5, mutation_rate=0.1, depot="0"):
    """
    Simple Genetic Algorithm to improve a population of solutions.
    Each individual is a dictionary with a 'segments' key (list of routes).
    """
    def flatten(individual):
        return [node for route in individual["segments"] for node in route]

    def crossover(parent1, parent2):
        flat1 = flatten(parent1)
        flat2 = flatten(parent2)
        cut = len(flat1) // 2
        child_flat = flat1[:cut] + [node for node in flat2 if node not in flat1[:cut]]
        return child_flat

    def mutate(route_flat, mutation_rate):
        for i in range(len(route_flat)):
            if random.random() < mutation_rate:
                j = random.randint(0, len(route_flat) - 1)
                route_flat[i], route_flat[j] = route_flat[j], route_flat[i]
        return route_flat

    demand = {node: 0 for node in distance_matrix if node != depot}
    for individual in population:
        for route in individual["segments"]:
            for node in route:
                if node in demand:
                    demand[node] = 1  # just to reuse split_route_prins

    best = min(population, key=lambda x: route_cost(x["segments"], distance_matrix))
    for _ in range(generations):
        new_pop = []
        for _ in range(len(population)):
            p1, p2 = random.sample(population, 2)
            child_flat = crossover(p1, p2)
            child_flat = mutate(child_flat, mutation_rate)
            new = split_route_prins(child_flat, demand, capacity=100, distance_matrix=distance_matrix, depot=depot)
            new_pop.append(new)
        best = min(new_pop + [best], key=lambda x: route_cost(x["segments"], distance_matrix))
        population = new_pop
    return population






def extract_charging_scheme(route):
   
    return {"charging_points": [node for node in route if node.startswith("F")]}

def split_route_prins(tour, demand, capacity, distance_matrix, depot="0"):

    if not tour:
        return {"segments": [], "total_cost": 0, "feasible": True}
    if len(tour) != len(set(tour)):
        raise ValueError("Tour contains duplicate nodes")
    for node in tour:
        if node not in demand:
            raise ValueError(f"Node {node} not found in demand dictionary")
        if demand[node] < 0:
            raise ValueError(f"Negative demand for node {node}")
        if demand[node] > capacity:
            raise ValueError(f"Demand of node {node} exceeds capacity")
        if node not in distance_matrix or depot not in distance_matrix[node]:
            raise ValueError(f"Invalid distance_matrix entry for node {node}")

    n = len(tour)
    
    # Check if entire tour is feasible (single trip)
    total_demand = sum(demand.get(node, 0) for node in tour)
    if total_demand <= capacity:
        seg_cost = 0
        if tour:
            seg_cost += distance_matrix[depot][tour[0]]
            for k in range(n - 1):
                seg_cost += distance_matrix[tour[k]][tour[k + 1]]
            seg_cost += distance_matrix[tour[n - 1]][depot]
        return {
            "segments": [tour] if tour else [],
            "total_cost": seg_cost,
            "feasible": True
        }

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
            if len(segment) == 1:
                seg_cost = distance_matrix[depot][segment[0]] + distance_matrix[segment[0]][depot]
            else:
                seg_cost = distance_matrix[depot][segment[0]]
                for k in range(len(segment) - 1):
                    seg_cost += distance_matrix[segment[k]][segment[k + 1]]
                seg_cost += distance_matrix[segment[-1]][depot]

            if cost[i] + seg_cost < cost[j]:
                cost[j] = cost[i] + seg_cost
                pred[j] = i

    segments = []
    idx = n
    while idx > 0:
        i = pred[idx]
        segments.insert(0, tour[i:idx])
        idx = i

    feasible = all(sum(demand.get(node, 0) for node in seg) <= capacity for seg in segments)

    total_cost = 0
    for segment in segments:
        if segment:
            total_cost += distance_matrix[depot][segment[0]]
            for i in range(len(segment) - 1):
                total_cost += distance_matrix[segment[i]][segment[i + 1]]
            total_cost += distance_matrix[segment[-1]][depot]

    return {
        "segments": segments,
        "total_cost": total_cost,
        "feasible": feasible
    }

def route_cost(route_segments, distance_matrix, depot="0"):

    total_cost = 0
    for segment in route_segments:
        if not segment:
            continue
        total_cost += distance_matrix[depot][segment[0]]
        for i in range(len(segment) - 1):
            total_cost += distance_matrix[segment[i]][segment[i + 1]]
        total_cost += distance_matrix[segment[-1]][depot]
    return total_cost

def update_pheromones(pheromones, P, distance_matrix, evaporation_rate=0.5):

    for edge in pheromones:
        pheromones[edge] *= (1 - evaporation_rate)
    
    best_solution = min(P, key=lambda x: route_cost(x["segments"], distance_matrix))
    best_cost = route_cost(best_solution["segments"], distance_matrix)
    if best_cost == 0:
        return
    
    for segment in best_solution["segments"]:
        route = ["0"] + segment + ["0"]
        for i in range(len(route) - 1):
            edge = (route[i], route[i + 1])
            pheromones[edge] = pheromones.get(edge, 1.0) + 1.0 / best_cost

# ------------------------
# Main ACO Function
# ------------------------

def ant_colony_optimization(I, F, N, pheromones, heuristic, demand, capacity, distance_matrix, depot="0"):
 
    I_hat = ["0"] + I + F
    P = []
    k = 1

    while k <= N:
        r = []
        available_nodes = list(set(I_hat) - {"0"})

        i = random.choice(available_nodes)
        r.append(i)
        available_nodes.remove(i)

        while available_nodes:
            current_node = r[-1]
            j = roulette_wheel_selection(available_nodes, current_node, pheromones, heuristic)
            r.append(j)
            available_nodes.remove(j)

        x = split_route_prins(r, demand, capacity, distance_matrix, depot)
        x["segments"] = [two_opt(route, distance_matrix, depot) for route in x["segments"]]
        P.append(x)
        k += 1

    update_pheromones(pheromones, P, distance_matrix)
    P_best = min(P, key=lambda x: route_cost(x["segments"], distance_matrix))
    P_best["segments"] = [two_opt(route, distance_matrix, depot) for route in P_best["segments"]]
    
    ls = extract_charging_scheme([n for seg in P_best["segments"] for n in seg])

    ##not implemented yet
    return P_best["segments"], ls


def interaction(Q, O, P_best, s, pheromone_matrix):
    # Step 1: Extract the basic customer routing sequence from the best ant
    rc = extract_customer_sequence(P_best)  # e.g., [1, 2, 3, 4]

    # Step 2: Evaluate all charging schemes in Q ∪ O based on rc
    combined_QO = Q + O
    for charging_scheme in combined_QO:
        charging_scheme.fitness = evaluate_solution(rc, charging_scheme)

    # Step 3: Select the top N best charging schemes for next generation
    combined_QO.sort(key=lambda cs: cs.fitness)
    Q = combined_QO[:N]

    # Step 4: Determine the best charging scheme
    Q_best = Q[0]

    # Step 5: Combine rc and Q_best to generate a new solution
    s_prime = combine_route_and_charging(rc, Q_best)

    # Step 6: Evaluate feasibility and compare quality
    if is_electricity_feasible(s_prime) and s_prime.cost < s.cost:
        s = s_prime
        update_pheromone(pheromone_matrix, s)
    else:
        update_pheromone(pheromone_matrix, P_best)

    return Q, pheromone_matrix, s


# ------------------------
# Visualization
# ------------------------

def visualize_solution(best_solution, distance_matrix, charging_scheme, depot="0"):
  
    G = nx.DiGraph()
    all_nodes = set([depot])
    edges = []

    for segment in best_solution:
        route = [depot] + segment + [depot]
        all_nodes.update(route)
        for i in range(len(route) - 1):
            edges.append((route[i], route[i + 1]))

    G.add_nodes_from(all_nodes)
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, seed=42)

    node_colors = []
    for node in G.nodes():
        if node == depot:
            node_colors.append("green")
        elif node.startswith("F"):
            node_colors.append("blue")
        else:
            node_colors.append("orange")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, edgecolors='black')
    nx.draw_networkx_labels(G, pos)

    edge_labels = {(u, v): f"{distance_matrix[u][v]:.1f}" for u, v in G.edges if v in distance_matrix[u]}
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    if "charging_points" in charging_scheme:
        for node in charging_scheme["charging_points"]:
            if node in G.nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color="cyan", node_size=700, edgecolors='black')

    plt.title("Ant Colony Optimization Route with Charging Stations")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('vrp_solution.png')
    # plt.close()
    plt.show()

# ------------------------
# Example Usage
# ------------------------

if __name__ == "__main__":
    I = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    F = ["F1", "F2", "F3"]
    N = 5  # number of ants per iteration
    capacity = 100  # ensure some feasibility
    num_iterations = 100  # number of ACO iterations

    demand = {
        "1": 2, "2": 3, "3": 4, "4": 5, "5": 6,
        "6": 7, "7": 8, "8": 9, "9": 9, "10": 1,
        "F1": 0, "F2": 0, "F3": 0
    }

    nodes = ["0"] + I + F
    distance_matrix = {i: {j: 5 for j in nodes if i != j} for i in nodes}
    for i in nodes:
        for j in nodes:
            if i != j:
                try:
                    distance_matrix[i][j] = abs(int(i.strip("F")) - int(j.strip("F"))) + 1
                except ValueError:
                    pass

    pheromones = {(i, j): 1.0 for i in nodes for j in nodes if i != j}
    heuristic = {(i, j): 1.0 / distance_matrix[i][j] if distance_matrix[i][j] != 0 else 1.0 for i in nodes for j in nodes if i != j}

    best_overall_solution = None
    best_overall_cost = float('inf')
    best_charging_scheme = None

    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1} ===")
        best_solution, charging_scheme = ant_colony_optimization(I, F, N, pheromones, heuristic, demand, capacity, distance_matrix)
        cost = route_cost(best_solution, distance_matrix)
        print(f"Cost: {cost}")
        if cost < best_overall_cost:
            best_overall_solution = best_solution
            best_overall_cost = cost
            best_charging_scheme = charging_scheme

    print("\n=== Final Best Solution ===")
    print("Best Solution:", best_overall_solution)
    print("Total Cost:", best_overall_cost)
    print("Charging Scheme:", best_charging_scheme)

    visualize_solution(best_overall_solution, distance_matrix, best_charging_scheme)
