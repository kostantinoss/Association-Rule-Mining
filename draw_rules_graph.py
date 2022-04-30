import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def draw_graph(rules,draw_choice):

    G = nx.DiGraph()

    color_map = []
    final_node_sizes = []

    color_iter = 0

    NumberOfRandomColors = 100
    edge_colors_iter = np.random.rand(NumberOfRandomColors)

    node_sizes = {}     # larger rule-nodes imply larger confidence
    node_colors = {}    # darker rule-nodes imply larger lift
    
    for index, row in rules.iterrows():

        if color_iter >=  100:
            break
        
        color_of_rule = edge_colors_iter[color_iter]
        rule = row['rule']
        rule_id = row['rule_ID']
        confidence = row['confidence']
        lift = row['lift']
        itemset = row['itemset']
        hypothesis=row['hypothesis']
        conclusion=row['conclusion']
        
        G.add_nodes_from(["R"+str(rule_id)])

        node_sizes.update({"R"+str(rule_id): float(confidence)})

        node_colors.update({"R"+str(rule_id): float(lift)})
        
        for item in hypothesis:
            G.add_edge(str(item), "R"+str(rule_id), color=color_of_rule)

        for item in conclusion:
            G.add_edge("R"+str(rule_id), str(item), color=color_of_rule)

        color_iter += 1 % NumberOfRandomColors


    print("\t++++++++++++++++++++++++++++++++++++++++")
    print("\tNode size & color coding:")
    print("\t----------------------------------------")
    print("\t[Rule-Node Size]")
    print("\t\t5 : lift = max_lilft, 4 : max_lift > lift > 0.75*max_lift + 0.25*min_lift")
    print("\t\t3 : 0.75*max_lift + 0.25*min_lift > lift > 0.5*max_lift + 0.5*min_lift")
    print("\t\t2 : 0.5*max_lift + 0.5*min_lift > lift > 0.25*max_lift + 0.75*min_lift")
    print("\t\t1 : 0.25*max_lift + 0.75*min_lift > lift > min_lift")
    print("\t----------------------------------------")
    print("\t[Rule-Node Color]")
    print("\t\tpurple : conf > 0.9, blue : conf > 0.75, cyan : conf > 0.6, green  : default")
    print("\t----------------------------------------")
    print("\t[Movie-Nodes]")
    print("\t\tSize: 1, Color: yellow")
    print("\t----------------------------------------")

    max_lift = rules['lift'].max()
    min_lift = rules['lift'].min()

    base_node_size = 500
    
    for node in G:

        if str(node).startswith("R"): # these are the rule-nodes...
                
            conf = node_sizes[str(node)]
            lift = node_colors[str(node)]
            
            # rule-node sizes encode lift...
            if lift == max_lift:
                final_node_sizes.append(base_node_size*5*lift)

            elif lift > 0.75*max_lift + 0.25*min_lift:
                final_node_sizes.append(base_node_size*4*lift)

            elif lift > 0.5*max_lift + 0.5*min_lift:
                final_node_sizes.append(base_node_size*3*lift)

            elif lift > 0.25*max_lift + 0.75*min_lift:
                final_node_sizes.append(base_node_size*2*lift)

            else: # lift >= min_lift...
                final_node_sizes.append(base_node_size*lift)

            # rule-node colors encode confidence...
            if conf > 0.9:
                color_map.append('purple')

            elif conf > 0.75:
                color_map.append('blue')

            elif conf > 0.6:
                color_map.append('cyan')

            else: # lift > min_confidence...
                color_map.append('green')

        else: # these are the movie-nodes...
            color_map.append('yellow') 
            final_node_sizes.append(2*base_node_size)

    edges = G.edges()
    colors = [G[u][v]['color'] for u, v in edges]

    if draw_choice == 'c': #circular layout
        nx.draw_circular(G, edges=edges, node_size=final_node_sizes, node_color = color_map, edge_color=colors, font_size=8, with_labels=True)

    elif draw_choice == 'r': #random layout
        nx.draw_random(G, edges=edges, node_size=final_node_sizes, node_color = color_map, edge_color=colors, font_size=8, with_labels=True)

    else: #spring layout...
        pos = nx.spring_layout(G, k=16, scale=1)
        nx.draw(G, pos, edges=edges, node_size=final_node_sizes, node_color = color_map, edge_color=colors, font_size=8, with_labels=False)
        nx.draw_networkx_labels(G, pos)    

    plt.show()

    # discovering most influential and most influenced movies
    # within highest-lift rules...
    outdegree_rules_sequence = {}
    outdegree_movies_sequence = {}
    indegree_rules_sequence = {}
    indegree_movies_sequence = {}
    
    outdegree_sequence = nx.out_degree_centrality(G)
    indegree_sequence = nx.in_degree_centrality(G)

    for (node, outdegree) in outdegree_sequence.items():
        # Check if this is a rule-node
        if str(node).startswith("R"):
            outdegree_rules_sequence[node] = outdegree
        else:
            outdegree_movies_sequence[node] = outdegree
            
    for (node, indegree) in indegree_sequence.items():
        # Check if this is a rule-node
        if str(node).startswith("R"):
            indegree_rules_sequence[node] = indegree
        else:
            indegree_movies_sequence[node] = indegree

    max_outdegree_movie_node = max(outdegree_movies_sequence, key=outdegree_movies_sequence.get)
    max_indegree_movie_node = max(indegree_movies_sequence, key=indegree_movies_sequence.get)
    print("\tMost influential movie (i.e., of maximum outdegree) wrt involved rules: ",max_outdegree_movie_node)
    print("\tMost influenced movie (i.e., of maximum indegree) wrt involved rules: ",max_indegree_movie_node)
