import json
import itertools

f = open("C:\\Student Handout\\Input data\\level0.json")
data = json.load(f)
#print(data)

n_neighbourhoods = data['n_neighbourhoods']
neighborhoods = data['neighbourhoods']
restaurant = data['restaurants']['r0']['neighbourhood_distance']

#TO SOLVE LEVEL 0 USING NEAREST NEIGHBOR HEURISTIC
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
json_object = json.dumps(output, indent=2)
with open("level0_output.json", "w") as outfile:
    outfile.write(json_object)

#LEVEL 0 using HELD KARP
n_neighbourhoods = data['n_neighbourhoods']
neighborhoods = data['neighbourhoods']
restaurant_distances = data['restaurants']['r0']['neighbourhood_distance']
#print(neighborhoods)

# Create the distance matrix
dist_matrix = []

# Add distances from restaurant to neighborhoods
dist_matrix.append([0] + restaurant_distances)
# Add distances between neighborhoods
for i in range(n_neighbourhoods):
    dist_row = [neighborhoods[f'n{i}']['distances'][j] for j in range(n_neighbourhoods)]
    dist_row.insert(0,restaurant_distances[i])
    #dist_row.append(n_neighbourhoods[i])
    dist_matrix.append(dist_row)


#Print the matrix for verification
#for row in dist_matrix:
#    print(row)
n = n_neighbourhoods + 1
memo = [[-1] * (1 << n) for i in range(n)]
next_node = [[-1] * (1 << n) for i in range(n)]

def tsp(i, mask):
    if mask == ((1 << i) | 3):
        return dist_matrix[0][i]

    if memo[i][mask] != -1:
        return memo[i][mask]

    res = float('inf')
    for j in range(1, n):
        if mask & (1 << j) and j != i:
            cur_res = tsp(j, mask & (~(1 << i))) + dist_matrix[j][i]
            if cur_res < res:
                res = cur_res
                next_node[i][mask] = j

    memo[i][mask] = res
    return res

# Find the minimum cost
min_cost = float('inf')
last_node = -1
final_mask = (1 << n) - 1

for i in range(0, n):
    cost = tsp(i, final_mask) + dist_matrix[i][0]
    if cost < min_cost:
        min_cost = cost
        last_node = i

# Reconstruct the path
path = [0]
# Assuming the rest of the code is the same as provided previously

# Initialize the starting point for path reconstruction
current_node = last_node
current_mask = final_mask

path = [current_node]  # Start from the last node
print(current_node)
# Reconstruct the path
while current_node > 0:
    # Get the next node (rename the variable to avoid conflict)
    next_node_in_path = next_node[current_node][current_mask]
    path.append(next_node_in_path)
    
    # Update the current mask by removing the current node from the set
    current_mask = current_mask & ~(1 << current_node)
    
    # Update the current node to the next node for the next iteration
    current_node = next_node_in_path

# Reverse the path to get the correct order starting from the restaurant
path = path[::-1]

# Include the restaurant as 'r0' at the beginning and end of the path
formatted_path = ['r0'] + [f'n{i-1}' for i in path[1:] if i != 0] + ['r0']

# Format the output
output = {"v0": {"path": formatted_path}}
output_json = json.dumps(output, indent=2)
print("Optimal path:", output_json)
with open("level0_output.json", "w") as outfile:
    outfile.write(json_object)