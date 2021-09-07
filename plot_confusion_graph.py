import networkx as nx
import matplotlib.pyplot as plt

if __name__ == '__main__':
    confusions = [['3','8','B','R'],
                  ['I','1','T','L'],
                  ['7','T'],
                  ['A','4'],
                  ['0','O','Q','D'],
                  ['Z','2','7'],
                  ['G','6'],
                  ['G','C'],
                  ['M','N'],
                  ['Y','V'],
                  ['V','W'],
                  ['U','J'],
                  ['K','X'],
                  ['E','F'],
                  ['5','S','9']]
    #Create confusion map with every character as key and
    #its possible confussions as values
    conf_map = {}
    for conf in confusions:
        for i in range(len(conf)):
            values = []
            key = conf[i]
            for j in range(len(conf)):
                if j != i:
                    values.append(conf[j])
            if key in conf_map:
                conf_map[key] += values
            else:
                conf_map[key] = values
    DG = nx.DiGraph()
    DG.add_nodes_from(conf_map.keys())
    for c in conf_map:
        for ch in conf_map[c]:
            DG.add_edge(c,ch)
    nx.draw(DG, with_labels=True, font_weight='bold')
    plt.show()
