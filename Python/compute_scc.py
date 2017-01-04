import collections


class Node:
    def __repr__(self):
        return "Node: %s" %(self.id)

    def __init__(self, graph, id):
        self.seen = False
        self.id = id
        graph.nodes[id] = self
        self.incoming = []
        self.outgoing = []

    @staticmethod
    def get_or_create(graph, id):
        return graph.nodes[id] if id in graph.nodes else Node(graph, id)

    @property
    def next(self):
        for edge in self.outgoing:
            if not edge.head.seen:
                return edge.head
        return None

class Edge:

    def __repr__(self):
        return "Edge: %s --> %s" %(self.tail, self.head)

    def __init__(self, graph, head, tail):
        self.head = Node.get_or_create(graph, head)
        self.tail = Node.get_or_create(graph, tail)
        self.head.incoming.append(self)
        self.tail.outgoing.append(self)

class Graph:

    def __repr__(self):
        return "Graph: [%s, ...., %s]" %(self.edges[0], self.edges[-1])

    def __init__(self):
        self.edges = []
        self.nodes = {}

    def reverse(self):
        Grev = Graph()
        for edge in self.edges:
            edge_reversed = Edge(Grev, edge.tail.id, edge.head.id)
            Grev.edges.append(edge_reversed)
        return Grev

    def reset(self):
        for id, node in self.nodes.items(): node.seen = False

G = Graph()
with open('SCC.txt', 'r') as f:
    for line in f:
        values = line.split(' ')
        head_id = int(values[1].strip())
        tail_id = int(values[0].strip())
        if head_id != tail_id:
            edge = Edge(G, head_id, tail_id)
            G.edges.append(edge)

Grev = G.reverse()
finishing_times = []
scc_sizes = []

print('First DFS')
sor = reversed(collections.OrderedDict(sorted(G.nodes.items())))
for id in sor:
    node = Grev.nodes[id]
    if not node.seen:
        to_explore = [node]
        while len(to_explore) != 0:
            curr = to_explore[-1]
            curr.seen = True
            if curr.next:
                to_explore.append(curr.next)
            else:
                # Reached end of DFS, append to finishing times
                n = to_explore.pop(-1)
                finishing_times.append(n.id)

print('Second DFS')
# Second DFS on G
while len(finishing_times) != 0:
    node = G.nodes[finishing_times.pop()]
    if not node.seen:
        scc_size = 0
        to_explore = [node]
        while len(to_explore) != 0:
            curr = to_explore[-1]
            curr.seen = True
            if curr.next:
                to_explore.append(curr.next)
            else:
                scc_size += 1
                n = to_explore.pop(-1)
        print('SCC :: Leader: %s, Size: %s' %(node, scc_size))
        scc_sizes.append(scc_size)

scc_sizes.sort()
print(scc_sizes, sum(scc_sizes))
