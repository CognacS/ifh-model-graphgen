import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
cimport cython

DTYPE = np.int32

@cython.boundscheck(False)
@cython.wraparound(False)
def get_bfs_order(int[:, :] edge_index, int n, int start):
    # n : number of vertices

    # get number of edges
    cdef int m = edge_index.shape[1]

    # 1 - compute outdegree of each vertex
    cdef int* outdegree = <int*>malloc(n * sizeof(int))
    for i in range(n):
        outdegree[i] = 0

    for i in range(m):
        outdegree[edge_index[0, i]] += 1

    # 2 - allocate memory for adjacency list
    cdef int **adj_list = <int**>malloc(n * sizeof(int *))
    for i in range(n):
        adj_list[i] = <int*>malloc(outdegree[i] * sizeof(int))

    # 3 - fill adjacency list
    for i in range(n):
        outdegree[i] = 0

    cdef int src
    cdef int dst
    for i in range(m):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        adj_list[src][outdegree[src]] = dst
        outdegree[src] += 1

    # ############ perform bfs ############

    # 1 - initialize queue
    cdef int *queue = <int*>malloc(n * sizeof(int))
    cdef int queue_start = 0
    cdef int queue_end = 0

    # 2 - initialize visited array
    # element i is 1 if vertex i has been visited, 0 otherwise
    cdef int *visited = <int*>malloc(n * sizeof(int))
    for i in range(n):
        visited[i] = 0

    # 3 - perform bfs
    queue[queue_end] = start
    queue_end = 1
    visited[start] = 1

    # initialize order as a numpy array
    cdef cnp.ndarray order_np = np.zeros(n, dtype=DTYPE)
    cdef int[:] order = order_np
    cdef int order_index = 0
    cdef int current, neighbor
    while order_index < n:
        # complete queue
        while queue_start < queue_end:
            # perform dequeue
            current = queue[queue_start]
            queue_start += 1
            # append current to order array
            order[order_index] = current
            order_index += 1
            # enqueue neighbors
            for i in range(outdegree[current]):
                neighbor = adj_list[current][i]
                if not visited[neighbor]:
                    queue[queue_end] = neighbor
                    queue_end += 1
                    visited[neighbor] = 1

        # safety check for disconnected graphs
        if order_index < n:
            for i in range(n):
                if not visited[i]:
                    queue[queue_end] = i
                    queue_end += 1
                    visited[i] = 1
                    break


    # ############ free memory ############
    free(outdegree)
    for i in range(n):
        free(adj_list[i])
    free(adj_list)
    free(queue)
    free(visited)

    return order_np