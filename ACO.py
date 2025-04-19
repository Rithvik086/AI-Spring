import random

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

def two_opt(route):
    return route  # Placeholder for now

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
            j = roulette_wheel_selection(available_nodes, current_node, pheromones, heuristic)
            r.append(j)
            available_nodes.remove(j)

        x = split_route_prins(r, demand, capacity, distance_matrix)
        x = [two_opt(route) for route in x]
        P.append(x)
        k += 1

    P_best = min(P, key=route_cost)
    P_best = [two_opt(route) for route in P_best]
    ls = extract_charging_scheme([n for seg in P_best for n in seg])

    return P_best, ls

# ------------------------
# Example Usage
# ------------------------

if __name__ == "__main__":
    I = ["1", "2", "3", ]
    F = ["F1", ]
    N = 5
    capacity = 4
    demand = {"1": 2, "2": 3, "3": 4,  "F1": 0, "F2": 0}

    nodes = ["0"] + I + F
    distance_matrix = {i: {} for i in nodes}
    for i in nodes:
        for j in nodes:
            if i != j:
                distance_matrix[i][j] = abs(int(i.strip("F")) - int(j.strip("F"))) + 1 if i.strip("F").isdigit() and j.strip("F").isdigit() else 5

    pheromones = {(i, j): 1.0 for i in nodes for j in nodes if i != j}
    heuristic = {(i, j): 1.0 / distance_matrix[i][j] for i in nodes for j in nodes if i != j}

    best_solution, charging_scheme = ant_colony_optimization(I, F, N, pheromones, heuristic, demand, capacity, distance_matrix)
    print("Best Solution:", best_solution)
    print("Charging Scheme:", charging_scheme)
