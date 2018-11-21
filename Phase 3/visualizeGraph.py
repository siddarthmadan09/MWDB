import matplotlib.pyplot as plt
import networkx as nx

def drawGraph(G):
    print (len(G.nodes))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'))
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color='r', arrows=True)
    plt.show()
