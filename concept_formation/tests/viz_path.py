import math
import graphviz
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import colorsys
from random import random
from random import choices
from collections import Counter, defaultdict

def get_color(p, vmin=0.0, vmax=1.0):
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.RdYlGn
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    mm = m.to_rgba(p)  # Use 'p' instead of 'x'
    M = colorsys.rgb_to_hsv(mm[0], mm[1], mm[2])
    return M


def prob_children_given_instance(node, instance, missing=False):
    probs = []
    s = 0
    for c in node.children:
        p_c = c.count / node.count

        if missing:
            log_prob = c.log_prob_instance_missing(instance)
        else:
            log_prob = c.log_prob_instance(instance)

        probs.append(p_c * math.exp(log_prob))
        s += probs[-1]

    return [p/s for p in probs]

def plot_best_search(instance, tree, max_nodes, greedy=False, consider_missing=False, obj=0):
    d = graphviz.Digraph()

    root = True
    if consider_missing:
        ll_instance = tree.root.log_prob_instance_missing(instance)
    else:
        ll_instance = tree.root.log_prob_instance(instance)
    ll_instance_root = ll_instance
    queue = [(tree.root, ll_instance, 0.0)]
    visited = []
    pred = defaultdict(Counter)

    leaves = 0

    obj0 = []
    obj1 = []
    obj2 = []
    obj3 = []


    while len(queue) > 0:
        curr, curr_ll_inst, curr_ll = queue.pop()

        if greedy:
            queue = []

        if obj == 0:
            score = curr_ll
            w = math.exp(score) 
        elif obj == 1:
            score = curr_ll_inst + curr_ll
            w = math.exp(score)
        elif obj == 2:
            score = math.exp(curr_ll) * (math.exp(curr_ll_inst) - math.exp(ll_instance_root))
            # w = score
            w = max(0, score)
        elif obj == 3:
            score = math.exp(curr_ll) * (curr_ll_inst - ll_instance_root)
            # w = score
            w = max(0, score)

        # print("Expanding node with score: ", score)
        # print("Weight: ", w)
        p = curr.predict_probs()
        # print("P(ctx=queen): ", p['context']['queen'])
        for attr in p:
            for val in p[attr]:
                pred[attr][val] += w * p[attr][val]

        if score is not None:
            visited.append((w, curr))
            obj0.append(math.exp(curr_ll))
            obj1.append(math.exp(curr_ll_inst + curr_ll))
            obj2.append(math.exp(curr_ll) * (math.exp(curr_ll_inst) - math.exp(ll_instance_root)))
            obj3.append(math.exp(curr_ll) * (curr_ll_inst - ll_instance_root))
        else:
            score = float('-inf')

        # visited.append((curr_ll_path, curr))
        leaves += 1
        # color = get_color(score)
        # color = get_color(math.exp(curr_ll_inst + curr_ll_path))
        is_leaf = "TERMINAL\n" if len(curr.children) == 0 else ""
        d.node(curr.concept_hash(), is_leaf + '{}\nscore: {:.10e}\nw: {:.10f}\nll(c|x)+ll(x|c): {:.10f}\nll(c|x): {:.10f}\nll(x|c): {:.10f}'.format(curr.concept_hash(), score, w, curr_ll_inst + curr_ll, curr_ll, curr_ll_inst),
               # color="%f, %f, %f" % (color[0], color[1], color[2]),
               style='filled')
        if root:
            root = False
        else:
            d.edge(curr.parent.concept_hash(), curr.concept_hash())

        if leaves >= max_nodes:
            break

        if len(curr.children) > 0:
            # children_probs = curr.prob_children_given_instance(instance)
            # print()
            # print(children_probs)
            # print(prob_children_given_instance(curr, instance))
            children_probs = prob_children_given_instance(curr, instance, missing=consider_missing)


            for i, c in enumerate(curr.children):
                ll_c = math.log(children_probs[i]) + curr_ll

                if consider_missing:
                    ll_instance = c.log_prob_instance_missing(instance)
                else:
                    ll_instance = c.log_prob_instance(instance)

                queue.append((c, ll_instance, ll_c))

            if obj == 0:
                queue.sort(key=lambda x: x[2])
            elif obj == 1:
                queue.sort(key=lambda x: x[1] + x[2])
            elif obj == 2:
                queue.sort(key=lambda x: math.exp(x[2]) * (math.exp(x[1]) - math.exp(ll_instance_root)))
            elif obj == 3:
                queue.sort(key=lambda x: math.exp(x[2]) * (x[1] - ll_instance_root))

    scores = [s for s, _ in visited]
    # plt.hist(scores)
    # plt.show()

    df = pd.DataFrame({'p(c|x)': obj0, 'p(c|x)*p(x|c)': obj1, 'p(c|x)*[p(x|c) - p(x)]': obj2, 'p(c|x)*[log p(x|c) - log p(x)]': obj3})
    # pd.plotting.scatter_matrix(df, alpha=0.2, diagonal="kde")
    axes = pd.plotting.scatter_matrix(df, alpha=0.5, diagonal='kde')
    corr = df.corr().to_numpy()
    for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
        axes[i, j].annotate("%.3f" %corr[i,j], (0.8, 0.8), xycoords='axes fraction', ha='center', va='center')
    plt.show()

    max_score = max(scores)
    min_score = min(scores)
    for score, node in visited:
        color = get_color(score, vmin=min_score, vmax=max_score)
        d.node(node.concept_hash(), color="%f, %f, %f" % (color[0], color[1], color[2]))

    for attr in pred:
        s = 0
        for val in pred[attr]:
            s += pred[attr][val]
        # print('norm', attr, 'by', s)
        for val in pred[attr]:
            pred[attr][val] /= s

    print("# LEAVES:", leaves)
    d.render(directory='graph_output', view=True)
    return pred

def plot_iterative_sampling(instance, tree, attr, target, n_samples, obj=0):
    d = graphviz.Digraph()

    visited = []
    visit_counts = Counter()
    leaves = 0

    for i in range(n_samples):
        root = True
        target_p = tree.root.predict_probs()[attr][target]
        ll_instance = tree.root.log_prob_instance(instance)
        ll_instance_root = tree.root.log_prob_instance(instance)
        queue = [(tree.root, ll_instance, 0.0, 0.0, target_p)]

        while len(queue) > 0:
            curr, curr_ll_inst, curr_ll, curr_ll_parent, curr_t_prob = choices(queue, weights=[math.exp(x[2]) for x in queue], k=1)[0]
            queue = []

            visit_counts[curr.concept_hash()] += 1

            if obj == 0:
                score = curr_ll
            elif obj == 1:
                if curr_ll == 0.0:
                    score = None
                else:
                    score = curr_ll_inst + curr_ll

            if score is not None:
                # visited.append((score, curr))
                visited.append((visit_counts[curr.concept_hash()], curr))
            else:
                score = float('-inf')

            # visited.append((curr_ll_path, curr))
            leaves += 1
            # color = get_color(score)
            # color = get_color(math.exp(curr_ll_inst + curr_ll_path))
            is_leaf = "TERMINAL\n" if len(curr.children) == 0 else ""
            d.node(curr.concept_hash(), is_leaf + '{}\nscore: {:.10f}\nvisits: {}\nll(c|x): {:.10f}\nll(x|c): {:.10f}\nTarget_prob: {:.10f}'.format(curr.concept_hash(), score, visit_counts[curr.concept_hash()], curr_ll, curr_ll_inst, curr_t_prob),
                   # color="%f, %f, %f" % (color[0], color[1], color[2]),
                   style='filled')

            if root:
                root = False
            else:
                d.edge(curr.parent.concept_hash(), curr.concept_hash())

            if len(curr.children) > 0:
                children_probs = curr.prob_children_given_instance(instance)

                for i, c in enumerate(curr.children):
                    ll_c = math.log(children_probs[i]) + curr_ll
                    target_p = c.predict_probs()[attr][target]
                    ll_instance = c.log_prob_instance(instance)
                    queue.append((c, ll_instance, ll_c, curr_ll, target_p))

                if obj == 0:
                    queue.sort(key=lambda x: x[2])
                elif obj == 1:
                    queue.sort(key=lambda x: x[1] + x[2])

    scores = [s for s, _ in visited]
    plt.hist(scores)
    plt.show()
    max_score = max(scores)
    min_score = min(scores)
    for score, node in visited:
        color = get_color(score, vmin=min_score, vmax=max_score)
        d.node(node.concept_hash(), color="%f, %f, %f" % (color[0], color[1], color[2]))

    print("# LEAVES:", leaves)
    d.render(directory='graph_output', view=True)

def plot_fringe(instance, tree, attr, target, max_expanded):
    d = graphviz.Digraph()

    root = True
    target_p = tree.root.predict_probs()[attr][target]
    ll_instance = tree.root.log_prob_instance(instance)
    queue = [(tree.root, ll_instance, 0.0, target_p)]
    terminals = []
    expanded = 0

    d.node(tree.root.concept_hash(), '{}\nscore: {:.10f}\nll(c|x)+ll(x|c): {:.10f}\nll(c|x): {:.10f}\nll(x|c): {:.10f}\nTarget_prob: {:.10f}'.format(tree.root.concept_hash(), 1.0, ll_instance, 0.0, ll_instance, target_p), style='filled')

    while len(queue) > 0 and max_expanded - expanded > 0:
        curr, curr_ll_inst, curr_ll, curr_t_prob = queue.pop()
        score = curr_ll
        expanded += 1

        if len(curr.children) == 0:
            terminals.append((curr_ll, curr))
        else:
            children_probs = curr.prob_children_given_instance(instance)

            for i, c in enumerate(curr.children):
                ll_c = math.log(children_probs[i]) + curr_ll
                target_p = c.predict_probs()[attr][target]
                ll_instance = c.log_prob_instance(instance)
                queue.append((c, ll_instance, ll_c, target_p))

                is_leaf = "TERMINAL\n" if len(c.children) == 0 else ""
                d.node(c.concept_hash(), is_leaf + '{}\nscore: {:.10f}\nll(c|x)+ll(x|c): {:.10f}\nll(c|x): {:.10f}\nll(x|c): {:.10f}\nTarget_prob: {:.10f}'.format(curr.concept_hash(), math.exp(ll_c), ll_instance + ll_c, ll_c, ll_instance, target_p), style='filled')
                d.edge(curr.concept_hash(), c.concept_hash())

            queue.sort(key=lambda x: x[2])

    scores = [math.exp(s) for s, _ in terminals] + [math.exp(x[2]) for x in queue]
    print(sum(scores))
    plt.hist(scores)
    plt.show()
    max_score = max(scores)
    min_score = min(scores)
    pred = defaultdict(Counter)
    for score, node in terminals:
        p = node.predict_probs()
        for attr in p:
            for val in p[attr]:
                pred[attr][val] += math.exp(score) * p[attr][val]
        color = get_color(math.exp(score), vmin=min_score, vmax=max_score)
        d.node(node.concept_hash(), color="%f, %f, %f" % (color[0], color[1], color[2]))

    for x in queue:
        node = x[0]
        score = x[2]
        p = node.predict_probs()
        for attr in p:
            for val in p[attr]:
                pred[attr][val] += math.exp(score) * p[attr][val]
        color = get_color(math.exp(score), vmin=min_score, vmax=max_score)
        d.node(node.concept_hash(), color="%f, %f, %f" % (color[0], color[1], color[2]))

    print("# EXPANDED:", expanded)
    d.render(directory='graph_output', view=True)

    return pred

