import csv
import json

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

    #towers = {}
    towers = []
    file_name = '40-student-datashop-no-endstate.txt'
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
            tower['action'] = action['action']

            if action['from'] == 'Inventory':
                block = "".join([i for i in row[key['ActionSel']] if not
                                 i.isdigit()])
                tower['r1'] = ["parameter1", "action", "inventory-" + block]

            else:
                tower['r1'] = ['parameter1', 'action', row[key['ActionSel']]]
            
            if action['to'] == 'Inventory':
                tower['r2'] = ["parameter2", "action", "inventory"]
            else:
                tower['destination'] = {}
                #tower['destination']['type'] = action['to']['type']
                tower['destination']['x'] = action['to']['x']
                tower['destination']['y'] = action['to']['y']
                tower['destination']['rotation'] = action['to']['rotation']
                tower['r2'] = ['parameter2', 'action', 'destination']

            remove = []

            # create integer representation?
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

            towers.append(tower)
            #towers[row[key['Step Name']]] = tower
            
        print("Removed %i blocks." % removal_count)

    #towers = [towers[s] for s in towers]

    with open('instant-test-processed2.json', 'w') as f:
        f.write(json.dumps(towers))
    


