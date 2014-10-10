# This is going to be the root file that you would use to work with each of the
# algorithms in this package. I went with tlc because of Trestle, Labyrinth, Cobweb
# but we can rename it later if we want

import json
from trestle import Trestle


def cluster(instances, depth=1):
    """
    Used to cluster examples incrementally and return the cluster labels.
    The final cluster labels are at a depth of 'depth' from the root. This
    defaults to 1, which takes the first split, but it might need to be 2
    or greater in cases where more distinction is needed.
    """
    tree = Trestle()


    temp_clusters = [tree.ifit(instance) for instance in instances]

    clusters = []
    for i,c in enumerate(temp_clusters):
        while (c.parent and c not in c.parent.children):
            c = c.parent
        while (isinstance(c, Trestle) and c.children):
            c = c.trestle_categorize_leaf(instances[i])

        promote = True
        while c.parent and promote:

            promote = True
            n = c
            for i in range(depth+2):
                if not n:
                    promote = False
                    break
                n = n.parent

            if promote:
                c = c.parent

        clusters.append(c.concept_name)

        while c:
            if not '_id' in c.av_counts:
                c.av_counts['_id'] = {}
            c.av_counts['_id'][i] = 1;
            c = c.parent

    with open('visualize/output.json', 'w') as f:
        f.write(json.dumps(tree.output_json()))

    return clusters
