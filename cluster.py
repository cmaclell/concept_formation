import json
import re

def cluster(tree, instances, minsplit=1, maxsplit=1, mod=True):
	"""
	Categorize the list of instances into the tree and return a list of lists of
	cluster labels.

	The size of the list will be (1 + maxsplit - minsplit). The first clustering
	is dervied by splitting the root, then each subsequent clustering is based
	on the splitting least coupled cluster (in terms of category utility). This
	may stop early if you reach the case where there are no more clusters to
	split (each instance in its own cluster).

	Keyword arguments:
	tree -- a CobwebTree, Cobweb3Tree, or TrestleTree
	instances -- a list of instance objects
	minsplit -- The minimum number of splits to perform on the tree, must be >=1
	(default 1)
	maxsplit -- the maximum number of splits to perform on the tree, must be >=1
	(default 1)
	mod -- If True instances will be fit (i.e. modyfiying knoweldge) else they
	will be categorized (i.e. not modiyfing knowledge) (default True)
	"""
	if minsplit < 1:
		raise ValueError("minsplit must be >= 1")
	if maxsplit < 1:
		raise ValueError("maxsplit must be >= 1")
	
	if mod:
		temp_clusters = [tree.ifit(instance) for instance in instances]
	else:
		temp_clusters = [tree.categorize(instance) for instance in instances]

	clusterings = []
	
	for nth_split in range(minsplit, maxsplit+1):

		clusters = []
		for i,c in enumerate(temp_clusters):
			while (c.parent and c.parent.parent):
				c = c.parent
			clusters.append("Concept" + c.concept_id)
		clusterings.append(clusters)

		split_cus = sorted([(tree.root.cu_for_split(c) -
							 tree.root.category_utility(), i, c) for i,c in
							enumerate(tree.root.children) if c.children])

		# Exit early, we don't need to re-reun the following part for the
		# last time through
		if nth_split == maxsplit or not split_cus:
			break

		# Split the least cohesive cluster
		tree.root.split(split_cus[-1][2])

	return clusterings

def generate_d3_visualization(tree, fileName):
	"""
	Export a .js file that is used to visualize the tree with d3.
	"""
	fname = 'visualize/'+fileName+'.js'
	with open(fname, 'w') as f:
		f.write("var output = '"+re.sub("'", '',
										json.dumps(tree.root.output_json()))+"';")

def depth_labels(tree,instances,mod=True):
	"""
	Categorize the list of instances into the tree and return a matrix of
	hierarchical labeling of each instance from the root to its most specific
	concept.

	The returned matrix is max(conceptDepth) X len(instances). Labelings are
	ordered general to specific with final_labels[0] being the root and
	final_labels[-1] being the leaves.
   
	Keyword attributes:
	tree -- a CobwebTree, Cobweb3Tree, or TrestleTree
	instances -- a list of instance objects
	mod -- If True instances will be fit (i.e. modyfiying knoweldge) else they
	will be categorized (i.e. not modiyfing knowledge) (default True)
	"""

	if mod:
		temp_labels = [tree.ifit(instance) for instance in instances]
	else:
		temp_labels = [tree.categorize(instance) for instance in instances]

	instance_labels = []
	max_depth = 0
	for t in temp_labels:
		labs = []
		depth = 0
		label = t
		while label.parent:
			labs.append("Concept" + label.concept_id)
			depth += 1
			label = label.parent
		labs.append("Concept" + label.concept_id)
		depth += 1
		instance_labels.append(labs)
		if depth > max_depth:
			max_depth = depth
   
	for f in instance_labels:
	   f.reverse()
	   last_label = f[-1]
	   while len(f) < max_depth:
		   f.append(last_label)
	

	final_labels = []
	for d in range(len(instance_labels[0])):
		depth_n = []
		for i in instance_labels:
			depth_n.append(i[d])
		final_labels.append(depth_n)

	return final_labels