import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq

class Graph:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.graph = {}

    def add_node(self, node):
        (x, y) = node
        self.graph[(x, y)] = []

    def add_edge(self, node1, node2, cost):
        self.graph[node1].append((node2, cost))
        self.graph[node2].append((node1, cost))

    def dijkstra(self, start, end):
        h = []
        visited = set()
        heapq.heappush(h, (0, start, []))
        while len(h) > 0:
            (cost, curr_node, path) = heapq.heappop(h)
            if curr_node in visited:
                continue
            if curr_node == end:
                path.append(curr_node)
                return path
            visited.add(curr_node)
            path.append(curr_node)
            for (neigh, edge_cost) in self.graph[curr_node]:
                if neigh not in visited:
                    heapq.heappush(h, (cost + edge_cost, neigh, path[:]))
        return []

def get_neighbors(img, node):
    (x, y) = node
    neighbors = []
    for i in range(x-1, x+2):
        for j in range(y-1, y+2):
            if i == x and j == y:
                continue
            if i < 0 or i >= img.shape[0] or j < 0 or j >= img.shape[1]:
                continue
            if img[i][j] == 255:
                continue
            neighbors.append((i, j))
    return neighbors

def get_shortest_path(img, start, end):
    rows, cols = img.shape
    graph = Graph(rows, cols)
    for i in range(rows):
        for j in range(cols):
            if img[i][j] == 0:
                node = (i, j)
                graph.add_node(node)
                neighbors = get_neighbors(img, node)
                for neigh in neighbors:
                    dist = np.sqrt((node[0] - neigh[0])*2 + (node[1] - neigh[1])*2)
                    graph.add_edge(node, neigh, dist)

    path = graph.dijkstra(start, end)
    if not path:
        return [], -1

    # Compute clearance for each node on the path
    clearance_matrix = []
    max_clearance = 0
    for node in path:
        clearance = 0
        for i in range(rows):
            for j in range(cols):
                if img[i][j] == 255:
                    dist = np.sqrt((node[0] - i)*2 + (node[1] - j)*2)
                    if dist < clearance or clearance == 0:
                        clearance = dist
        clearance_matrix.append((node, clearance))
        if clearance > max_clearance:
            max_clearance = clearance

    return clearance_matrix, max_clearance

# Load the image
img = cv2.imread('map.png', cv2.IMREAD_GRAYSCALE)
print(img.shape)

# Get user input for start and end nodes
start_x = int(input("Enter x coordinate of start node: "))
start_y = int(input("Enter y coordinate of start node: "))

end_x = int(input("Enter x coordinate of end node: "))
end_y = int(input("Enter y coordinate of end node: "))

start = (start_x, start_y)
end = (end_x, end_y)
dist = get_shortest_path(img, start, end)
print(f"Shortest path distance: {dist}")