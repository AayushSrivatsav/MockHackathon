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

with open("level1_output.json", "w") as outfile:
    outfile.write(output_json)