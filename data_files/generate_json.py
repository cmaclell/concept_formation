import csv
import json
import copy

def approx_match(b1, b2):
    if 'x' not in b1 or 'x' not in b2:
        return False
    if 'y' not in b1 or 'y' not in b2:
        return False
    if 'rotation' not in b1 or 'rotation' not in b2:
        return False
    if 'type' not in b1 or 'type' not in b2:
        return False

    if b1['type'] != b2['type']:
        return False

    if not abs(b1['x'] - b2['x']) < 10:
        return False
    if not abs(b1['y'] - b2['y']) < 10:
        return False
    #if not abs(b1['rotation'] - b2['rotation']) < 20.0:
    #    return False

    return True

if __name__ == "__main__":

    towers = {}
    tower_actions = {}
    #towers = []
    file_name = 'instant-test-processed.txt'
    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        previous = None
        key = {}

        removal_count = 0
        for row in reader:
            if not key:
                for i,v in enumerate(row):
                    key[v] = i
                continue

            tower = json.loads(row[key['Selection']])
            #tower['guid'] = row[key['guid']]
            tower['guid'] = row[key['Step Name']]
            tower['Problem Name'] = row[key['Problem Name']]
            tower['Outcome'] = row[key['Outcome']]

            #if row[key['Action']] == 'End_State':
            #    tower['success'] = row[key['Outcome']]
            #else:
            #    del tower['UFO']
            action = json.loads(row[key['Input']])
            action_relation = []
            action_relation.append(action['action'])

            #tower['action'] = action['action']

            if action['from'] == 'Inventory':
                block = action['to']['type']
                #block = "".join([i for i in row[key['ActionSel']] if not
                #                 i.isdigit()])
                action_relation.append("inventory-" + block)
                #tower['r1'] = ["parameter1", "action", "inventory-" + block]

            else:
                #TODO do matching to get the correct name from the state!
                for b in tower:
                    if approx_match(tower[b], action['from']):
                        block = b
                action_relation.append(block)

                #tower['r1'] = ['parameter1', 'action', row[key['ActionSel']]]
                #action_relation.append(row[key['ActionSel']])
            
            if action['to'] == 'Inventory':
                block = action['from']['type']
                #block = "".join([i for i in row[key['ActionSel']] if not
                #                 i.isdigit()])
                action_relation.append("inventory-" + block)
                #tower['r2'] = ["parameter2", "action", "inventory"]
            else:
                tower['destination'] = {}
                tower['destination']['type'] = action['to']['type']
                tower['destination']['x'] = action['to']['x']
                tower['destination']['y'] = action['to']['y']
                tower['destination']['rotation'] = action['to']['rotation']
                action_relation.append('destination')

                #tower['r2'] = ['parameter2', 'action', 'destination']

            remove = []

            # TODO could create integer representation here
            for name in tower:
                if isinstance(tower[name], dict):
                    for v in tower[name]:
                        if isinstance(tower[name][v], float):
                            #tower[name][v] = int(tower[name][v])
                            if tower[name][v] < 0:
                                remove.append(name)
            if remove:
                removal_count += len(remove)
                for name in remove:
                    del tower[name]

            tower['action'] = action_relation
            if tower['guid'] not in tower_actions:
                tower_actions[tower['guid']] = set()
            tower_actions[tower['guid']].add(tuple(action_relation))

            towers[tower['guid']] = tower
            #towers.append(tower)
            #towers[row[key['Step Name']]] = tower
            
        print("Removed %i blocks." % removal_count)

    #towers = [towers[s] for s in towers]

    tower_action_pairs = []
    for guid in tower_actions:
        for action in tower_actions[guid]:
            tower = copy.deepcopy(towers[guid])
            tower['action']
            tower_action_pairs.append(tower)


    with open('instant-test-processed.json', 'w') as f:
        f.write(json.dumps(tower_action_pairs))
    


