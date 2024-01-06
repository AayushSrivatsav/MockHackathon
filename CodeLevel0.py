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
for row in dist_matrix:
    print(row)
n = 20
memo = [[-1]*(1 << (n+1)) for i in range(n+1)]
 
 
def fun(i, mask):
    # base case
    # if only ith bit and 1st bit is set in our mask,
    # it implies we have visited all other nodes already
    if mask == ((1 << i) | 3):
        return dist_matrix[1][i]
 
    # memoization
    if memo[i][mask] != -1:
        return memo[i][mask]
 
    res = 10**9  # result of this sub-problem
 
    # we have to travel all nodes j in mask and end the path at ith node
    # so for every node j in mask, recursively calculate cost of 
    # travelling all nodes in mask
    # except i and then travel back from node j to node i taking 
    # the shortest path take the minimum of all possible j nodes
    for j in range(1, n+1):
        if (mask & (1 << j)) != 0 and j != i and j != 1:
            res = min(res, fun(j, mask & (~(1 << i))) + dist_matrix[j][i])
    memo[i][mask] = res  # storing the minimum value
    return res
 
 
# Driver program to test above logic
ans = 10**9
for i in range(1, n+1):
    # try to go from node 1 visiting all nodes in between to i
    # then return from i taking the shortest route to 1
    ans = min(ans, fun(i, (1 << (n+1))-1) + dist_matrix[i][1])
 
print("The cost of most efficient tour = " + str(ans))