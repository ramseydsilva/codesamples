
class Node:
    def __repr__(self):
        return "Node: %s" %(self.id)

    def __init__(self, graph, id):
        self.id = id
        graph.nodes[id] = self
        self.neighbours = {}

    @staticmethod
    def get_or_create(graph, id):
        return graph.nodes[id] if id in graph.nodes else Node(graph, id)

    def add_neighbour(self, neighbour, edge):
        if neighbour not in self.neighbours:
            self.neighbours[neighbour.id] = edge

class Edge:

    def __repr__(self):
        return "Edge: %s --> %s" %(self.tail, self.head)

    def __init__(self, graph, head, tail, weight):
        self.seen = False
        self.weight = weight
        self.head = Node.get_or_create(graph, head)
        self.tail = Node.get_or_create(graph, tail)
        self.head.add_neighbour(self.tail, self)
        self.tail.add_neighbour(self.head, self)
        graph.edges.append(self)

class Graph:

    def __repr__(self):
        if len(self.edges) > 2:
            return "Graph: [%s, ...., %s]" %(self.edges[0], self.edges[-1])
        return "Graph"

    def __len__(self):
        return len(self.nodes)

    def __init__(self):
        self.edges = []
        self.nodes = {}

    def copy(self):
        new_graph = Graph()
        new_graph.edges = self.edges
        new_graph.nodes = self.nodes
        return new_graph

    def move_to(self, graph, node_id):
        node = self.nodes[node_id]
        graph.nodes[node_id] = node
        for neighbour_id, edge in node.neighbours.items():
            if edge.seen:
                if edge.tail in graph.nodes:
                    edge.seen = False
        del self.nodes[node_id]

V = Graph()
with open('dijkstraData.txt', 'r') as f:
    for line in f:
        values = line.split(' ')
        head_id = int(values[0].strip())
        for v in values[1:]:
            if ',' in v:
                tail_id, weight = v.split(',')
                edge = Edge(V, int(head_id), int(tail_id), int(weight))

# Initial empty Graph
X = Graph()
# Initial full graph
V_minus_X = V.copy()

# Move Node 1 from V to X
V_minus_X.move_to(X, 1)
GS = {1: 0}

print(len(X), len(V_minus_X))

while len(V_minus_X) != 0:
    CE = []
    selected_node = None
    min_gs = 0
    for node_id, node in X.nodes.items():
        for neighbour_id, edge in node.neighbours.items():
            if neighbour_id not in X.nodes:
                CE.append(CE)
                tail = None
                head = None
                if edge.tail.id == node_id:
                    head = edge.tail.id
                    tail = edge.head.id
                else:
                    head = edge.head.id
                    tail = edge.tail.id

                gs = GS[head] + edge.weight
                if gs < min_gs or min_gs == 0:
                    selected_node = tail
                    min_gs = gs

    print('Selected node %s from V to X' %(selected_node))
    GS[selected_node] = min_gs
    V_minus_X.move_to(X, selected_node)
    print('Moved node %s from V to X' %(selected_node))

print(len(X), len(V_minus_X))
print("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" %(GS[7], GS[37], GS[59], GS[82], GS[99], GS[115], GS[133], GS[165], GS[188], GS[197]))
