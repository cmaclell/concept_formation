import math
import graphviz
import matplotlib as mpl
import matplotlib.cm as cm
import colorsys

def get_color(p):
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = cm.RdYlGn
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    mm = m.to_rgba(p)  # Use 'p' instead of 'x'
    M = colorsys.rgb_to_hsv(mm[0], mm[1], mm[2])
    return M

def plot_path_probs(instance, tree, attr, target):
    d = graphviz.Digraph()

    target_p = tree.root.predict_probs()[attr][target]
    color = get_color(1.0)
    d.node(tree.root.concept_hash(), 'CHOSEN\n{}\nConcept prob: {:.2f}\nPath prob: {:.10f}\nTarget prob: {:.10f}'.format(tree.root.concept_hash(), 1.0, 1.0, target_p), color="%f, %f, %f" % (color[0], color[1], color[2]), style='filled')
    queue = [(tree.root, 1.0, True)]

    while len(queue) > 0:
        curr, path_prob, curr_chosen = queue.pop()
        # if path_prob < 0.1 and not curr_chosen:
        #     continue
        children_probs = curr.prob_children_given_instance(instance)
        max_prob = 0.0
        if len(children_probs) > 0:
            max_prob = max(children_probs)
        for i, c in enumerate(curr.children):
            c_chosen = curr_chosen and children_probs[i] == max_prob
            c_path_prob = children_probs[i] * path_prob
            target_p = c.predict_probs()[attr][target]
            if c_path_prob < 0.01 and not c_chosen:
                continue
            color = get_color(c_path_prob)
            chosen_str = "CHOSEN\n" if c_chosen else ""
            d.node(c.concept_hash(),
                   chosen_str + '{}\nConcept prob: {:.2f}\nPath prob: {:.10f}\nTarget_prob: {:.10f}'.format(
                       c.concept_hash(), children_probs[i], c_path_prob, target_p),
                   color="%f, %f, %f" % (color[0], color[1], color[2]), style='filled')
            d.edge(curr.concept_hash(), c.concept_hash())
            queue.append((c, c_path_prob, c_chosen))

    d.render(directory='graph_output', view=True)

def plot_astar_paths(instance, tree, attr, target):
    d = graphviz.Digraph()

    root = True
    target_p = tree.root.predict_probs()[attr][target]
    queue = [(tree.root, 1.0, 1.0, target_p)]

    while len(queue) > 0:
        curr, curr_c_prob, curr_p_prob, curr_t_prob = queue.pop()
        color = get_color(curr_p_prob)
        is_leaf = "TERMINAL\n" if len(curr.children) == 0 else ""
        d.node(curr.concept_hash(), is_leaf + '{}\nConcept prob: {:.2f}\nPath prob: {:.10f}\nTarget_prob: {:.10f}'.format(curr.concept_hash(), curr_c_prob, curr_p_prob, curr_t_prob), color="%f, %f, %f" % (color[0], color[1], color[2]), style='filled')
        if root:
            root = False
        else:
            d.edge(curr.parent.concept_hash(), curr.concept_hash())

        if len(curr.children) == 0:
            break

        children_probs = curr.prob_children_given_instance(instance)
        max_prob = 0.0
        if len(children_probs) > 0:
            max_prob = max(children_probs)

        for i, c in enumerate(curr.children):
            c_path_prob = children_probs[i] * curr_p_prob
            target_p = c.predict_probs()[attr][target]
            queue.append((c, children_probs[i], c_path_prob, target_p))

        queue.sort(key=lambda x: x[1])

    d.render(directory='graph_output', view=True)


def plot_frontier_paths(instance, tree, attr, target, max_nodes):
    d = graphviz.Digraph()

    root = True
    target_p = tree.root.predict_probs()[attr][target]
    ll_instance = tree.root.log_prob_instance(instance)
    queue = [(tree.root, ll_instance, 0.0, target_p)]

    leaves = 0

    while len(queue) > 0:
        curr, curr_ll_inst, curr_ll_path, curr_t_prob = queue.pop()
        leaves += 1
        color = get_color(curr_t_prob)
        # color = get_color(math.exp(curr_ll_inst + curr_ll_path))
        is_leaf = "TERMINAL\n" if len(curr.children) == 0 else ""
        d.node(curr.concept_hash(), is_leaf + '{}\nll concept: {:.10f}\nll path: {:.10f}\nll inst: {:.10f}\nTarget_prob: {:.10f}'.format(curr.concept_hash(), curr_ll_inst + curr_ll_path, curr_ll_path, curr_ll_inst, curr_t_prob), color="%f, %f, %f" % (color[0], color[1], color[2]), style='filled')
        if root:
            root = False
        else:
            d.edge(curr.parent.concept_hash(), curr.concept_hash())

        if leaves >= max_nodes:
            break

        if len(curr.children) > 0:
            children_probs = curr.prob_children_given_instance(instance)
            max_prob = 0.0
            if len(children_probs) > 0:
                max_prob = max(children_probs)

            for i, c in enumerate(curr.children):
                ll_path = math.log(children_probs[i]) + curr_ll_path
                target_p = c.predict_probs()[attr][target]
                ll_instance = c.log_prob_instance(instance)
                queue.append((c, ll_instance, ll_path, target_p))

            queue.sort(key=lambda x: x[1] + x[2])

    print("# LEAVES:", leaves)
    d.render(directory='graph_output', view=True)
    

def plot_beam(width, instance, tree, attr, target):
    d = graphviz.Digraph()

    root = True
    target_p = tree.root.predict_probs()[attr][target]
    fringe = [(tree.root, 1.0, 1.0, target_p)]

    while len(fringe) > 0:
        queue = []
        while len(fringe) > 0:
            curr, curr_c_prob, curr_p_prob, curr_t_prob = fringe.pop()
            color = get_color(curr_p_prob)
            is_leaf = "TERMINAL\n" if len(curr.children) == 0 else ""
            d.node(curr.concept_hash(), is_leaf + '{}\nConcept prob: {:.2f}\nPath prob: {:.10f}\nTarget_prob: {:.10f}'.format(curr.concept_hash(), curr_c_prob, curr_p_prob, curr_t_prob), color="%f, %f, %f" % (color[0], color[1], color[2]), style='filled')
            if root:
                root = False
            else:
                d.edge(curr.parent.concept_hash(), curr.concept_hash())

            children_probs = curr.prob_children_given_instance(instance)
            max_prob = 0.0
            if len(children_probs) > 0:
                max_prob = max(children_probs)

            for i, c in enumerate(curr.children):
                c_path_prob = children_probs[i] * curr_p_prob
                target_p = c.predict_probs()[attr][target]
                queue.append((c, children_probs[i], c_path_prob, target_p))

        queue.sort(key=lambda x: x[1])
        fringe = queue[-width:]

    d.render(directory='graph_output', view=True)
