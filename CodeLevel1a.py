'''
import json

# Load the data
file_path = "C:\Student Handout\Input data\level1a.json"

with open(file_path, 'r') as file:
    data = json.load(file)

# Extract data
n_neighbourhoods = data['n_neighbourhoods']
neighborhoods = data['neighbourhoods']
vehicle_capacity = data['vehicles']['v0']['capacity']
restaurant_distances = data['restaurants']['r0']['neighbourhood_distance']
orders = {neighborhood: details['order_quantity'] for neighborhood, details in neighborhoods.items()}

dist_matrix = []

# Add distances from restaurant to neighborhoods
dist_matrix.append([0] + restaurant_distances)
# Add distances between neighborhoods
for i in range(n_neighbourhoods):
    dist_row = [neighborhoods[f'n{i}']['distances'][j] for j in range(n_neighbourhoods)]
    dist_row.insert(0,restaurant_distances[i])
    #dist_row.append(n_neighbourhoods[i])
    dist_matrix.append(dist_row)


# Function to find the nearest neighborhood
def find_nearest(current_location, unvisited, distances):
    nearest = None
    min_distance = float('inf')
    for n in unvisited:
        if distances[current_location][int(n[1:])] < min_distance:
            min_distance = distances[current_location][int(n[1:])]
            nearest = n
    return nearest

# Create delivery slots
delivery_slots = []
current_slot = []
current_capacity = 0

for neighborhood, details in neighborhoods.items():
    order_quantity = details['order_quantity']
    if current_capacity + order_quantity <= vehicle_capacity:
        current_slot.append(neighborhood)
        current_capacity += order_quantity
    else:
        delivery_slots.append(current_slot)
        current_slot = [neighborhood]
        current_capacity = order_quantity

if current_slot:
    delivery_slots.append(current_slot)

# Optimize each delivery slot with a simple TSP solution (Nearest Neighbor)
optimized_slots = []
for slot in delivery_slots:
    route = ['r0']  # Start at the restaurant
    unvisited = set(slot)
    while unvisited:
        current_location = int(route[-1][1:]) if route[-1] != 'r0' else 0
        next_stop = find_nearest(current_location, unvisited, dist_matrix)
        route.append(next_stop)
        unvisited.remove(next_stop)
    route.append('r0')  # Return to the restaurant
    optimized_slots.append(route)

output = {"v0": {}}
for i, slot in enumerate(optimized_slots, start=1):
    output["v0"][f"path{i}"] = slot

# Convert to JSON format
output_json = json.dumps(output, indent=2)
print("Optimized Delivery Slots in JSON format:")
print(output_json)

with open("level1a_output.json", "w") as outfile:
    outfile.write(output_json)

#BINPACKING AND VEHICLE ROUTING
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
def first_fit_decreasing(orders, capacity):
    # Sort the orders in decreasing order by quantity
    sorted_orders = sorted(orders, key=lambda x: x[1], reverse=True)

    bins = []
    for order in sorted_orders:
        placed = False
        for bin in bins:
            if sum([o[1] for o in bin]) + order[1] <= capacity:
                bin.append(order)
                placed = True
                break

        if not placed:
            bins.append([order])

    return bins
#bins = first_fit_decreasing(orders, vehicle_capacity)
def solve_vrp(neighborhoods, dist_matrix, vehicle_capacity):
    delivery_slots = []
    current_slot = []
    current_capacity = 0

    for neighborhood, details in neighborhoods.items():
        order_quantity = details['order_quantity']
        if current_capacity + order_quantity <= vehicle_capacity:
            current_slot.append(neighborhood)
            current_capacity += order_quantity
        else:
            delivery_slots.append(current_slot)
            current_slot = [neighborhood]
            current_capacity = order_quantity

    if current_slot:
        delivery_slots.append(current_slot)

    optimized_slots = {}
    for i, slot in enumerate(delivery_slots, start=1):
        route = ['r0']
        unvisited = set(slot)
        while unvisited:
            current_location = int(route[-1][1:]) if route[-1] != 'r0' else 0
            next_stop = find_nearest(current_location, unvisited, dist_matrix)
            route.append(next_stop)
            unvisited.remove(next_stop)
        route.append('r0')
        optimized_slots[f'path{i}'] = route

    return optimized_slots

optimized_routes = solve_vrp(neighborhoods, dist_matrix, vehicle_capacity)

# Format the output
output = {"v0": optimized_routes}
output_json = json.dumps(output, indent=2)
print("Optimized Delivery Routes:")
print(output_json)
with open("level1a_output.json", "w") as outfile:
    outfile.write(output_json)


import json

def first_fit_decreasing(orders, capacity):
    sorted_orders = sorted(orders, key=lambda x: x[1], reverse=True)
    bins = []
    for order in sorted_orders:
        placed = False
        for bin in bins:
            if sum(o[1] for o in bin) + order[1] <= capacity:
                bin.append(order)
                placed = True
                break
        if not placed:
            bins.append([order])
    return bins

def find_nearest(current_location, unvisited, distances):
    nearest = None
    min_distance = float('inf')
    for n in unvisited:
        dist = distances[current_location][int(n[1:])]
        if dist < min_distance:
            min_distance = dist
            nearest = n
    return nearest

def solve_vrp(orders, dist_matrix, vehicle_capacity):
    bins = first_fit_decreasing(orders, vehicle_capacity)
    optimized_routes = {}

    for i, bin in enumerate(bins, start=1):
        route = ['r0']
        unvisited = set(n[0] for n in bin)
        while unvisited:
            current_location = int(route[-1][1:]) if route[-1] != 'r0' else 0
            next_stop = find_nearest(current_location, unvisited, dist_matrix)
            route.append(next_stop)
            unvisited.remove(next_stop)
        route.append('r0')
        optimized_routes[f'path{i}'] = route

    return optimized_routes

# Load the data
file_path = "path_to_your_file.json"
with open(f"C:\Student Handout\Input data\level1a.json", 'r') as file:
    data = json.load(file)

# Parse data
n_neighbourhoods = data['n_neighbourhoods']
neighborhoods = data['neighbourhoods']
restaurant_distances = data['restaurants']['r0']['neighbourhood_distance']
vehicle_capacity = data['vehicles']['v0']['capacity']

# Create distance matrix
dist_matrix = [[0] * (n_neighbourhoods + 1) for _ in range(n_neighbourhoods + 1)]
dist_matrix[0] = [0] + restaurant_distances
for i in range(n_neighbourhoods):
    dist_matrix[i + 1] = [restaurant_distances[i]] + neighborhoods[f'n{i}']['distances']

# Create orders list
orders = [(f'n{i}', neighborhoods[f'n{i}']['order_quantity']) for i in range(n_neighbourhoods)]

# Solve VRP
optimized_routes = solve_vrp(orders, dist_matrix, vehicle_capacity)

# Format and output results
output = {"v0": optimized_routes}
output_json = json.dumps(output, indent=2)
print(output_json)
with open("level1a_output.json", "w") as outfile:
    outfile.write(output_json)


import json
import itertools

def first_fit_decreasing(orders, capacity):
    sorted_orders = sorted(orders, key=lambda x: x[1], reverse=True)
    bins = []
    for order in sorted_orders:
        placed = False
        for bin in bins:
            if sum(o[1] for o in bin) + order[1] <= capacity:
                bin.append(order)
                placed = True
                break
        if not placed:
            bins.append([order])
    return bins

def calculate_total_distance(route, dist_matrix):
    total_dist = 0
    for i in range(len(route) - 1):
        total_dist += dist_matrix[int(route[i][1:])][int(route[i+1][1:])]
    return total_dist

def find_optimal_route(slot, dist_matrix):
    min_route = None
    min_distance = float('inf')
    for perm in itertools.permutations(slot):
        current_route = ['r0'] + list(perm) + ['r0']
        current_distance = calculate_total_distance(current_route, dist_matrix)
        if current_distance < min_distance:
            min_distance = current_distance
            min_route = current_route
    return min_route

def solve_vrp(orders, dist_matrix, vehicle_capacity):
    bins = first_fit_decreasing(orders, vehicle_capacity)
    optimized_routes = {}

    for i, bin in enumerate(bins, start=1):
        slot = [n[0] for n in bin]
        optimized_route = find_optimal_route(slot, dist_matrix)
        optimized_routes[f'path{i}'] = optimized_route

    return optimized_routes

# Load the data
file_path = "C:\Student Handout\Input data\level1a.json"
with open(file_path, 'r') as file:
    data = json.load(file)

# Parse data
n_neighbourhoods = data['n_neighbourhoods']
neighborhoods = data['neighbourhoods']
restaurant_distances = data['restaurants']['r0']['neighbourhood_distance']
vehicle_capacity = data['vehicles']['v0']['capacity']

# Create distance matrix
dist_matrix = [[0] * (n_neighbourhoods + 1) for _ in range(n_neighbourhoods + 1)]
dist_matrix[0] = [0] + restaurant_distances
for i in range(n_neighbourhoods):
    dist_matrix[i + 1] = [restaurant_distances[i]] + neighborhoods[f'n{i}']['distances']

# Create orders list
orders = [(f'n{i}', neighborhoods[f'n{i}']['order_quantity']) for i in range(n_neighbourhoods)]

# Solve VRP
optimized_routes = solve_vrp(orders, dist_matrix, vehicle_capacity)

# Format and output results
output = {"v0": optimized_routes}
output_json = json.dumps(output, indent=2)
print(output_json)
with open("level1a_output.json", "w") as outfile:
    outfile.write(output_json)

import json
import itertools

def first_fit_decreasing(orders, capacity):
    sorted_orders = sorted(orders, key=lambda x: x[1], reverse=True)
    bins = []
    for order in sorted_orders:
        placed = False
        for bin in bins:
            if sum(o[1] for o in bin) + order[1] <= capacity:
                bin.append(order)
                placed = True
                break
        if not placed:
            bins.append([order])
    return bins

def nearest_neighbor_route(slot, dist_matrix):
    route = ['r0']
    unvisited = set(slot)
    while unvisited:
        current_location = int(route[-1][1:]) if route[-1] != 'r0' else 0
        next_stop = min(unvisited, key=lambda x: dist_matrix[current_location][int(x[1:])])
        route.append(next_stop)
        unvisited.remove(next_stop)
    route.append('r0')
    return route

def two_opt_swap(route, dist_matrix):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue  # Skip adjacent edges
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                if calculate_total_distance(new_route, dist_matrix) < calculate_total_distance(best, dist_matrix):
                    best = new_route
                    improved = True
        route = best
    return best

def calculate_total_distance(route, dist_matrix):
    total_dist = 0
    for i in range(len(route) - 1):
        total_dist += dist_matrix[int(route[i][1:])][int(route[i+1][1:])]
    return total_dist

def solve_vrp(orders, dist_matrix, vehicle_capacity):
    bins = first_fit_decreasing(orders, vehicle_capacity)
    optimized_routes = {}

    for i, bin in enumerate(bins, start=1):
        slot = [n[0] for n in bin]
        initial_route = nearest_neighbor_route(slot, dist_matrix)
        optimized_route = two_opt_swap(initial_route, dist_matrix)
        optimized_routes[f'path{i}'] = optimized_route

    return optimized_routes

# Load the data
file_path = "c:\Student Handout\Input data\level1a.json"
with open(file_path, 'r') as file:
    data = json.load(file)

# Parse data
n_neighbourhoods = data['n_neighbourhoods']
neighborhoods = data['neighbourhoods']
restaurant_distances = data['restaurants']['r0']['neighbourhood_distance']
vehicle_capacity = data['vehicles']['v0']['capacity']

# Create distance matrix
dist_matrix = [[0] * (n_neighbourhoods + 1) for _ in range(n_neighbourhoods + 1)]
dist_matrix[0] = [0] + restaurant_distances
for i in range(n_neighbourhoods):
    dist_matrix[i + 1] = [restaurant_distances[i]] + neighborhoods[f'n{i}']['distances']

# Create orders list
orders = [(f'n{i}', neighborhoods[f'n{i}']['order_quantity']) for i in range(n_neighbourhoods)]

# Solve VRP
optimized_routes = solve_vrp(orders, dist_matrix, vehicle_capacity)

# Format and output results
output = {"v0": optimized_routes}
output_json = json.dumps(output, indent=2)
print(output_json)
with open("level1a_output.json", "w") as outfile:
    outfile.write(output_json)
    '''

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import json

def create_data_model(data):
    """Stores the data for the problem."""
    data_model = {}
    data_model['distance_matrix'] = create_distance_matrix(data)
    data_model['demands'] = [0] + [data['neighbourhoods'][f'n{i}']['order_quantity'] for i in range(data['n_neighbourhoods'])]
    data_model['vehicle_capacities'] = [data['vehicles']['v0']['capacity']]
    data_model['num_vehicles'] = 1
    data_model['depot'] = 0
    return data_model

def create_distance_matrix(data):
    """Creates the distance matrix from the data."""
    distances = [[0] * (data['n_neighbourhoods'] + 1) for _ in range(data['n_neighbourhoods'] + 1)]
    for i in range(data['n_neighbourhoods']):
        distances[0][i + 1] = data['restaurants']['r0']['neighbourhood_distance'][i]
        distances[i + 1][0] = data['restaurants']['r0']['neighbourhood_distance'][i]
        for j in range(data['n_neighbourhoods']):
            distances[i + 1][j + 1] = data['neighbourhoods'][f'n{i}']['distances'][j]
    return distances

def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    total_distance = 0
    total_load = 0
    routes = {}
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = ['r0']
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            if node_index != 0:
                plan_output.append(f'n{node_index - 1}')
        plan_output.append('r0')
        routes[f'path{vehicle_id + 1}'] = plan_output
        total_distance += route_distance
        total_load += route_load
    return routes

# Load the data
file_path = "c:\\Student Handout\\Input data\\level1a.json"
with open(file_path, 'r') as file:
    data = json.load(file)

# Instantiate the data problem.
data_model = create_data_model(data)

# Create the routing index manager.
manager = pywrapcp.RoutingIndexManager(len(data_model['distance_matrix']),
                                       data_model['num_vehicles'], data_model['depot'])

# Create Routing Model.
routing = pywrapcp.RoutingModel(manager)

# Define cost of each arc.
def distance_callback(from_index, to_index):
    """Returns the distance between the two nodes."""
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data_model['distance_matrix'][from_node][to_node]

transit_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# Add Capacity constraint.
def demand_callback(from_index):
    """Returns the demand of the node."""
    from_node = manager.IndexToNode(from_index)
    return data_model['demands'][from_node]

demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
routing.AddDimensionWithVehicleCapacity(
    demand_callback_index,
    0,  # null capacity slack
    data_model['vehicle_capacities'],  # vehicle maximum capacities
    True,  # start cumul to zero
    'Capacity')

# Setting first solution heuristic.
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

# Solve the problem.
solution = routing.SolveWithParameters(search_parameters)

# Print solution on console.
if solution:
    routes = print_solution(data_model, manager, routing, solution)
    output = {"v0": routes}
    print(json.dumps(output, indent=2))
else:
    print('No solution found !')

