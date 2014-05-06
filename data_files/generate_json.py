import csv
import json

if __name__ == "__main__":

    towers = {}
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

            if row[key['Action']] == 'End_State':
                tower['success'] = row[key['Outcome']]
            else:
                del tower['UFO']


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

            towers[row[key['Step Name']]] = tower
            
        print("Removed %i blocks." % removal_count)

    towers = [towers[s] for s in towers]

    with open('instant-test-processed.json', 'w') as f:
        f.write(json.dumps(towers))
    


