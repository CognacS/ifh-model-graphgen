from collections import deque
import numpy as np

def get_bfs_order(edge_index, n, start):

    neighborhood = [[] for _ in range(n)]
    for i in range(edge_index.shape[1]):
        neighborhood[edge_index[0][i]].append(edge_index[1][i])

    visited = [False] * n

    queue = deque([start])
    visited[start] = True
    order = [0] * n
    order_index = 0
    while order_index < n:
        # complete queue
        while queue:
            current = queue.popleft()
            order[order_index] = current
            order_index += 1
            for neighbor in neighborhood[current]:
                if not visited[neighbor]:
                    queue.append(neighbor)
                    visited[neighbor] = True

        # safety check for disconnected graphs
        if order_index < n:
            for i in range(n):
                if not visited[i]:
                    queue.append(i)
                    visited[i] = True
                    break

    arr = np.array(order, dtype=np.int32)

    return arr