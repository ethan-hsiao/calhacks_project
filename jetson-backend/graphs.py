import numpy as np
from math import *
import matplotlib.pyplot as plt

class Graph(object):
    def __init__(self, edge_list=None, coords=None):
        """ initializes a graph object from either an edge list (priority)
            or an array of points 2d points (will connect everything together)
        """
        if edge_list and coords:
            self.edge_list = edge_list
            self.coords = coords
        elif coords:
            self.edge_list = self.gen_edge_list(coords)
            self.coords = coords
        else:
            raise Exception("Need to provide coords to create graph")

    def add_edge(self, edge):
        """ appends an edge to edge list"""
        self.edge_list.append(edge)

    def gen_edge_list(self, coords):
        """generates an edge list from unlabeled coordinates"""
        def distance(p1,p2):
            return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

        edge_list = []
        for i, p1 in enumerate(coords):
            for j, p2 in enumerate(coords):
                if i<j:
                    edge_list.append([i,j,distance(p1,p2)])
        return edge_list

    def get_adj_list(self):
        """return adj list representation of self"""
        adj_list = [[] for i in range(self.size())]

        for a,b,c in self.edge_list:
            adj_list[a].append([b,c])
            adj_list[b].append([a,c])

        return adj_list

    def get_max_length(self):
        adj_list = self.get_adj_list()
        visited = [False for i in range(self.size())]
        def dfs(a):
            if visited[a]:
                return 0
            visited[a] = True
            total = 1
            for b,w in adj_list[a]:
                total += dfs(b)
            return total
        lengths = [dfs(a) for a in range(self.size())]
        if lengths:
            return max(lengths)
        else:
            return 0

    def mst(self):
        """ returns an MST of self"""
        self.edge_list = sorted(self.edge_list, key = lambda x: x[2])

        new_edge_list = []

        link = [i for i in range(self.size())]
        size = [1 for i in range(self.size())]

        def find(x):
            while x!=link[x]:
                x = link[x]
            return x

        def same(a,b):
            return find(a)==find(b)

        def unite(a,b,c):
            new_edge_list.append([a,b,c])
            a = find(a)
            b = find(b)
            if (size[a]<size[b]):
                temp = a
                a = b
                b = temp
            size[a] += size[b]
            link[b] = a

        for a,b,c in self.edge_list:
            if not same(a,b):
                unite(a,b,c)

        return Graph(edge_list=new_edge_list, coords=self.coords)

    def prune_long_edges(self, z_score):
        """returns a list of graphs where edges with weight more than z_score std deviations away from mean have been removed"""
        lengths = np.array([c for a,b,c in self.edge_list])
        std = np.std(lengths)
        mean = np.mean(lengths)

        new_edge_list = [[a,b,c] for a,b,c in self.edge_list if (c-mean)/std < z_score]

        return Graph(edge_list=new_edge_list, coords=self.coords).part()

    def part(self):
        adj_list = self.get_adj_list()
        visited = [False for i in range(self.size())]

        graphs = []
        def dfs(a):
            visited[a] = True

            nodes_visited.append(a)
            for b,w in adj_list[a]:
                if not visited[b]:
                    edges_used.append([a,b,w])
                    dfs(b)

        for a in range(self.size()):
            if visited[a]:
                continue
            nodes_visited = []
            edges_used = []
            dfs(a)
            coords = [self.coords[i] for i in nodes_visited]

            d = {}
            index = 0
            new_edge_list = []
            for a,b,c in edges_used:
                if a not in d:
                    d[a] = index
                    index += 1
                a = d[a]
                if b not in d:
                    d[b] = index
                    index += 1
                b = d[b]
                new_edge_list.append([a,b,c])

            graphs.append(Graph(edge_list=new_edge_list,coords=coords))

        return graphs

    def faces_to_connections(faces):
        """faces - array of [[x0,y0,w0,h0],[x1,y1,w1,h1],...]
           connections - array [[x0,y0,x1,y1],...] where x0,y0 to x1,y1 connects the centers of faces
           from faces array with mst. ROUNDING to int
        """

        g = []
        for (x, y, w, h) in faces:
            g.append([x+w/2,y+h/2])
        graph = Graph(coords=g)

        c = []
        for a, connections in enumerate(graph.mst().get_adj_list()):
            for b,w in connections:
                c.append([int(faces[a][0]+faces[a][2]/2),int(faces[a][1]+faces[a][3]/2),int(faces[b][0]+faces[b][2]/2),int(faces[b][1]+faces[b][3]/2)])
        return c

    def size(self):
        """returns number of nodes"""
        s = 0
        for a,b,w in self.edge_list:
            s = max(max(a+1,b+1),s)
        return s

    def __str__(self):
        res = "edges:"
        for a,b,c in self.edge_list:
            res += '\n'+str(a)+'=>'+str(b)+'   weight = ' + str(round(c,2))
        return res
