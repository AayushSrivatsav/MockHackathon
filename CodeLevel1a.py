import json

# Load the data
file_path = "path_to_your_file/level0.json"

with open(file_path, 'r') as file:
    data = json.load(file)

# Extract data
n_neighbourhoods = data['n_neighbourhoods']
neighborhoods = data['neighbourhoods']
vehicle_capacity = data['vehicles']['v0']['capacity']
restaurant_distances = data['restaurants']['r0']['neighbourhood_distance']

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

# Output the optimized delivery slots
print("Optimized Delivery Slots:")
for slot in optimized_slots:
    print(slot)
