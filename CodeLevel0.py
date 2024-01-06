import json

f = open("C:\\Student Handout\\Input data\\level0.json")
data = json.load(f)
#print(data)

n_neighbourhoods = data['n_neighbourhoods']
neighborhoods = data['neighbourhoods']
restaurant = data['restaurants']['r0']['neighbourhood_distance']

def find_nearest(current_distances, visited):
    nearest = None
    min_distance = float('inf')
    for i in range(len(current_distances)):
        if i not in visited and current_distances[i] < min_distance:
            min_distance = current_distances[i]
            nearest = i
    return nearest

# Initialize variables
visited = set()
tour = ['r0']
current_location = 0  # Start at the restaurant
while len(visited) < n_neighbourhoods:
    next_location = find_nearest(neighborhoods[f"n{current_location}"]['distances'], visited)
    visited.add(next_location)
    tour.append(f"n{next_location}")
    current_location = next_location

# Add the return trip to the restaurant
tour.append('r0')

# Format the output
output = {"v0": {"path": tour}}
print(json.dumps(output, indent=2))