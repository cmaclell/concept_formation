def get_color(p):
    import matplotlib as mpl
    import matplotlib.cm as cm
    import colorsys
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = cm.RdYlGn
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    mm = m.to_rgba(p)  # Use 'p' instead of 'x'
    M = colorsys.rgb_to_hsv(mm[0], mm[1], mm[2])
    return M

def plot_path_probs(instance, tree, attr, target):
    import graphviz
    d = graphviz.Digraph()

    target_p = tree.root.predict_probs()[attr][target]
    color = get_color(1.0)
    d.node(tree.root.concept_hash(), 'CHOSEN\nConcept prob: {:.2f}\nPath prob: {:.2f}\nTarget prob: {:.2f}'.format(1.0, 1.0, target_p), color="%f, %f, %f" % (color[0], color[1], color[2]), style='filled')
    queue = [(tree.root, 1.0, True)]

    while len(queue) > 0:
        curr, path_prob, curr_chosen = queue.pop()
        children_probs = curr.prob_children_given_instance(instance)
        max_prob = 0.0
        if len(children_probs) > 0:
            max_prob = max(children_probs)
        for i, c in enumerate(curr.children):
            c_chosen = curr_chosen and children_probs[i] == max_prob
            c_path_prob = children_probs[i] * path_prob
            if c_path_prob < 1e-3 and not c_chosen:
                continue
            target_p = c.predict_probs()[attr][target]
            color = get_color(c_path_prob)
            chosen_str = "CHOSEN\n" if c_chosen else ""
            d.node(c.concept_hash(),
                   chosen_str + 'Concept prob: {:.2f}\nPath prob: {:.2f}\nTarget_prob: {:.2f}'.format(
                       children_probs[i], c_path_prob, target_p),
                   color="%f, %f, %f" % (color[0], color[1], color[2]), style='filled')
            d.edge(curr.concept_hash(), c.concept_hash())
            queue.append((c, c_path_prob, c_chosen))

    d.render(directory='graph_output', view=True)

