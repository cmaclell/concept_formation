from random import shuffle
from concept_formation.cobweb import CobwebTree
from concept_formation.visualize import visualize

data = []

data.append({'_item': 'one',
             '_super': 'pounder',
             '_basic': 'hammer',
             '_sub': 'hammer1',
             'handle': 'one',
             'shaft': 'one',
             'head': 'one',
             'size': 'one'})

data.append({'_item': 'two',
             '_super': 'pounder',
             '_basic': 'hammer',
             '_sub': 'hammer1',
             'handle': 'one',
             'shaft': 'one',
             'head': 'one',
             'size': 'two'})

data.append({'_item': 'three',
             '_super': 'pounder',
             '_basic': 'hammer',
             '_sub': 'hammer2',
             'handle': 'one',
             'shaft': 'one',
             'head': 'two',
             'size': 'one'})

data.append({'_item': 'four',
             '_super': 'pounder',
             '_basic': 'hammer',
             '_sub': 'hammer2',
             'handle': 'one',
             'shaft': 'one',
             'head': 'two',
             'size': 'two'})

data.append({'_item': 'five',
             '_super': 'pounder',
             '_basic': 'brick',
             '_sub': 'brick1',
             'handle': 'two',
             'shaft': 'two',
             'head': 'three',
             'size': 'one'})

data.append({'_item': 'six',
             '_super': 'pounder',
             '_basic': 'brick',
             '_sub': 'brick1',
             'handle': 'two',
             'shaft': 'two',
             'head': 'three',
             'size': 'two'})

data.append({'_item': 'seven',
             '_super': 'pounder',
             '_basic': 'brick',
             '_sub': 'brick2',
             'handle': 'three',
             'shaft': 'two',
             'head': 'three',
             'size': 'one'})

data.append({'_item': 'eight',
             '_super': 'pounder',
             '_basic': 'brick',
             '_sub': 'brick2',
             'handle': 'three',
             'shaft': 'two',
             'head': 'three',
             'size': 'two'})

data.append({'_item': 'nine',
             '_super': 'cutter',
             '_basic': 'knife',
             '_sub': 'knife1',
             'handle': 'four',
             'shaft': 'three',
             'head': 'four',
             'size': 'one'})

data.append({'_item': 'ten',
             '_super': 'cutter',
             '_basic': 'knife',
             '_sub': 'knife1',
             'handle': 'four',
             'shaft': 'three',
             'head': 'four',
             'size': 'two'})

data.append({'_item': 'eleven',
             '_super': 'cutter',
             '_basic': 'knife',
             '_sub': 'knife2',
             'handle': 'four',
             'shaft': 'three',
             'head': 'five',
             'size': 'one'})

data.append({'_item': 'twelve',
             '_super': 'cutter',
             '_basic': 'knife',
             '_sub': 'knife2',
             'handle': 'four',
             'shaft': 'three',
             'head': 'five',
             'size': 'two'})

data.append({'_item': 'thirteen',
             '_super': 'cutter',
             '_basic': 'pizza cutter',
             '_sub': 'pizza1',
             'handle': 'five',
             'shaft': 'four',
             'head': 'six',
             'size': 'one'})

data.append({'_item': 'fourteen',
             '_super': 'cutter',
             '_basic': 'pizza cutter',
             '_sub': 'pizza1',
             'handle': 'five',
             'shaft': 'four',
             'head': 'six',
             'size': 'two'})

data.append({'_item': 'fifteen',
             '_super': 'cutter',
             '_basic': 'pizza cutter',
             '_sub': 'pizza2',
             'handle': 'five',
             'shaft': 'five',
             'head': 'six',
             'size': 'one'})

data.append({'_item': 'sixteen',
             '_super': 'cutter',
             '_basic': 'pizza cutter',
             '_sub': 'pizza2',
             'handle': 'five',
             'shaft': 'five',
             'head': 'six',
             'size': 'two'})


tree = CobwebTree()

for i in range(1):
    shuffle(data)
    tree.fit(data)
visualize(tree)
