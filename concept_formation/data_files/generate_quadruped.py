from random import normalvariate
from random import choice

def bellrand(start, finish):
    """
    Meant to mimic Gennari's crude truncated normal.
    """
    mu = start + ((finish - start) / 2)
    std = (finish - start) / 2

    sample = float('inf')
    while sample < start or sample > finish:
        sample = normalvariate(mu, std)

    return sample

def generate_animals(num_instances):
    return [generate_random_animal() for i in range(num_instances)]

def generate_random_animal():
    genfun = [generate_dog, generate_cat, generate_horse, generate_giraffe]
    f = choice(genfun)
    return f()

def generate_dog():
    dog = {}
    dog['_type'] = 'dog'

    torso = 'torso'
    leg1 = 'leg1'
    leg2 = 'leg2'
    leg3 = 'leg3'
    leg4 = 'leg4'
    neck = 'neck'
    head = 'head'
    tail = 'tail'
    
    dog[torso] = {}
    dog[torso]['_type'] = 'torso'
    dog[torso]['locationX'] = 0
    dog[torso]['locationY'] = 0
    dog[torso]['locationZ'] = 0
    dog[torso]['axisX'] = 1
    dog[torso]['axisY'] = 0
    dog[torso]['axisZ'] = 0
    dog[torso]['height'] = bellrand(22.0, 28.0)
    dog[torso]['radius'] = bellrand(5.5, 7.5)
    dog[torso]['texture'] = bellrand(170, 180)

    leglength = bellrand(10, 15)

    dog[leg1] = {}
    dog[leg1]['_type'] = 'leg1'
    dog[leg1]['locationX'] = dog[torso]['height'] / 2
    dog[leg1]['locationY'] = dog[torso]['radius']
    dog[leg1]['locationZ'] = -dog[torso]['radius'] - leglength/2
    dog[leg1]['axisX'] = bellrand(0,1)
    dog[leg1]['axisY'] = 0
    dog[leg1]['axisZ'] = -1
    dog[leg1]['height'] = leglength
    dog[leg1]['radius'] = bellrand(0.8,1.2)
    dog[leg1]['texture'] = bellrand(170,180)

    dog[leg2] = {}
    dog[leg2]['_type'] = 'leg2'
    dog[leg2]['locationX'] = dog[torso]['height'] / 2
    dog[leg2]['locationY'] = -dog[torso]['radius']
    dog[leg2]['locationZ'] = -dog[torso]['radius'] - leglength/2
    dog[leg2]['axisX'] = 0
    dog[leg2]['axisY'] = 0
    dog[leg2]['axisZ'] = -1
    dog[leg2]['height'] = leglength
    dog[leg2]['radius'] = bellrand(0.8, 1.2)
    dog[leg2]['texture'] = bellrand(170, 180)

    dog[leg3] = {}
    dog[leg3]['_type'] = 'leg3'
    dog[leg3]['locationX'] = -dog[torso]['height'] / 2
    dog[leg3]['locationY'] = dog[torso]['radius']
    dog[leg3]['locationZ'] = -dog[torso]['radius'] - leglength/2
    dog[leg3]['axisX'] = 0
    dog[leg3]['axisY'] = 0
    dog[leg3]['axisZ'] = -1
    dog[leg3]['height'] = leglength
    dog[leg3]['radius'] = bellrand(0.8, 1.2)
    dog[leg3]['texture'] = bellrand(170, 180)

    dog[leg4] = {}
    dog[leg4]['_type'] = 'leg4'
    dog[leg4]['locationX'] = -dog[torso]['height'] / 2
    dog[leg4]['locationY'] = -dog[torso]['radius']
    dog[leg4]['locationZ'] = -dog[torso]['radius'] - leglength/2
    dog[leg4]['axisX'] = bellrand(0.0, 1.0)
    dog[leg4]['axisY'] = 0
    dog[leg4]['axisZ'] = -1
    dog[leg4]['height'] = leglength
    dog[leg4]['radius'] = bellrand(0.8, 1.2)
    dog[leg4]['texture'] = bellrand(170, 180)

    neck_height = bellrand(5.0, 6.0)

    dog[neck] = {}
    dog[neck]['_type'] = 'neck'
    dog[neck]['locationX'] = dog[torso]['height'] / 2 + neck_height / 2
    dog[neck]['locationY'] = 0
    dog[neck]['locationZ'] = dog[torso]['radius']
    dog[neck]['axisX'] = 1
    dog[neck]['axisY'] = 0
    dog[neck]['axisZ'] = 1
    dog[neck]['height'] = neck_height
    dog[neck]['radius'] = bellrand(2.0, 3.0)
    dog[neck]['texture'] = bellrand(170, 180)

    dog[head] = {}
    dog[head]['_type'] = 'head'
    dog[head]['locationX'] = dog[torso]['height'] / 2 + neck_height
    dog[head]['locationY'] = 0
    dog[head]['locationZ'] = dog[torso]['radius']
    dog[head]['axisX'] = 1
    dog[head]['axisY'] = bellrand(-1, 1)
    dog[head]['axisZ'] = 1
    dog[head]['height'] = bellrand(10, 13)
    dog[head]['radius'] = bellrand(2.5, 3.5)
    dog[head]['texture'] = bellrand(170, 180)

    tail_height = bellrand(6, 8)
    dog[tail] = {}
    dog[tail]['_type'] = 'tail'
    dog[tail]['locationX'] = -dog[torso]['height'] + tail_height
    dog[tail]['locationY'] = 0
    dog[tail]['locationZ'] = 0
    dog[tail]['axisX'] = -1
    dog[tail]['axisY'] = bellrand(-1, 1)
    dog[tail]['axisZ'] = bellrand(-1, 1)
    dog[tail]['height'] = tail_height
    dog[tail]['radius'] = bellrand(0.2, 0.4)
    dog[tail]['texture'] = bellrand(170, 180)

    return dog

def generate_cat():
    cat = {}
    cat['_type'] = 'cat'

    torso = 'torso'
    leg1 = 'leg1'
    leg2 = 'leg2'
    leg3 = 'leg3'
    leg4 = 'leg4'
    neck = 'neck'
    head = 'head'
    tail = 'tail'
    
    cat[torso] = {}
    cat[torso]['_type'] = 'torso'
    cat[torso]['locationX'] = 0
    cat[torso]['locationY'] = 0
    cat[torso]['locationZ'] = 0
    cat[torso]['axisX'] = 1
    cat[torso]['axisY'] = 0
    cat[torso]['axisZ'] = 0
    cat[torso]['height'] = bellrand(15.0, 22.0)
    cat[torso]['radius'] = bellrand(2.5, 4.5)
    cat[torso]['texture'] = bellrand(170, 180)

    leglength = bellrand(4, 9)

    cat[leg1] = {}
    cat[leg1]['_type'] = 'leg1'
    cat[leg1]['locationX'] = cat[torso]['height'] / 2
    cat[leg1]['locationY'] = cat[torso]['radius']
    cat[leg1]['locationZ'] = -cat[torso]['radius'] - leglength/2
    cat[leg1]['axisX'] = bellrand(0,1)
    cat[leg1]['axisY'] = 0
    cat[leg1]['axisZ'] = -1
    cat[leg1]['height'] = leglength
    cat[leg1]['radius'] = bellrand(0.4,0.8)
    cat[leg1]['texture'] = bellrand(170,180)

    cat[leg2] = {}
    cat[leg2]['_type'] = 'leg2'
    cat[leg2]['locationX'] = cat[torso]['height'] / 2
    cat[leg2]['locationY'] = -cat[torso]['radius']
    cat[leg2]['locationZ'] = -cat[torso]['radius'] - leglength/2
    cat[leg2]['axisX'] = 0
    cat[leg2]['axisY'] = 0
    cat[leg2]['axisZ'] = -1
    cat[leg2]['height'] = leglength
    cat[leg2]['radius'] = bellrand(0.4, 0.8)
    cat[leg2]['texture'] = bellrand(170, 180)

    cat[leg3] = {}
    cat[leg3]['_type'] = 'leg3'
    cat[leg3]['locationX'] = -cat[torso]['height'] / 2
    cat[leg3]['locationY'] = cat[torso]['radius']
    cat[leg3]['locationZ'] = -cat[torso]['radius'] - leglength/2
    cat[leg3]['axisX'] = 0
    cat[leg3]['axisY'] = 0
    cat[leg3]['axisZ'] = -1
    cat[leg3]['height'] = leglength
    cat[leg3]['radius'] = bellrand(0.4, 0.8)
    cat[leg3]['texture'] = bellrand(170, 180)

    cat[leg4] = {}
    cat[leg4]['_type'] = 'leg4'
    cat[leg4]['locationX'] = -cat[torso]['height'] / 2
    cat[leg4]['locationY'] = -cat[torso]['radius']
    cat[leg4]['locationZ'] = -cat[torso]['radius'] - leglength/2
    cat[leg4]['axisX'] = bellrand(0.0, 1.0)
    cat[leg4]['axisY'] = 0
    cat[leg4]['axisZ'] = -1
    cat[leg4]['height'] = leglength
    cat[leg4]['radius'] = bellrand(0.4, 0.8)
    cat[leg4]['texture'] = bellrand(170, 180)

    neck_height = bellrand(2, 4)

    cat[neck] = {}
    cat[neck]['_type'] = 'neck'
    cat[neck]['locationX'] = cat[torso]['height'] / 2 + neck_height / 2
    cat[neck]['locationY'] = 0
    cat[neck]['locationZ'] = cat[torso]['radius']
    cat[neck]['axisX'] = 1
    cat[neck]['axisY'] = 0
    cat[neck]['axisZ'] = 1
    cat[neck]['height'] = neck_height
    cat[neck]['radius'] = bellrand(1.5, 2.5)
    cat[neck]['texture'] = bellrand(170, 180)

    cat[head] = {}
    cat[head]['_type'] = 'head'
    cat[head]['locationX'] = cat[torso]['height'] / 2 + neck_height
    cat[head]['locationY'] = 0
    cat[head]['locationZ'] = cat[torso]['radius']
    cat[head]['axisX'] = 1
    cat[head]['axisY'] = bellrand(-1, 1)
    cat[head]['axisZ'] = 1
    cat[head]['height'] = bellrand(3, 5)
    cat[head]['radius'] = bellrand(1.5, 2.5)
    cat[head]['texture'] = bellrand(170, 180)

    tail_height = bellrand(10, 18)
    cat[tail] = {}
    cat[tail]['_type'] = 'tail'
    cat[tail]['locationX'] = -cat[torso]['height'] + tail_height
    cat[tail]['locationY'] = 0
    cat[tail]['locationZ'] = 0
    cat[tail]['axisX'] = -1
    cat[tail]['axisY'] = bellrand(-1, 1)
    cat[tail]['axisZ'] = bellrand(-1, 1)
    cat[tail]['height'] = tail_height
    cat[tail]['radius'] = bellrand(0.3, 0.7)
    cat[tail]['texture'] = bellrand(170, 180)

    return cat

def generate_horse():
    horse = {}
    horse['_type'] = 'horse'

    torso = 'torso'
    leg1 = 'leg1'
    leg2 = 'leg2'
    leg3 = 'leg3'
    leg4 = 'leg4'
    neck = 'neck'
    head = 'head'
    tail = 'tail'
    
    horse[torso] = {}
    horse[torso]['_type'] = 'torso'
    horse[torso]['locationX'] = 0
    horse[torso]['locationY'] = 0
    horse[torso]['locationZ'] = 0
    horse[torso]['axisX'] = 1
    horse[torso]['axisY'] = 0
    horse[torso]['axisZ'] = 0
    horse[torso]['height'] = bellrand(50, 60)
    horse[torso]['radius'] = bellrand(10, 14.5)
    horse[torso]['texture'] = bellrand(170, 180)

    leglength = bellrand(36, 44)

    horse[leg1] = {}
    horse[leg1]['_type'] = 'leg1'
    horse[leg1]['locationX'] = horse[torso]['height'] / 2
    horse[leg1]['locationY'] = horse[torso]['radius']
    horse[leg1]['locationZ'] = -horse[torso]['radius'] - leglength/2
    horse[leg1]['axisX'] = bellrand(0, 0.5)
    horse[leg1]['axisY'] = 0
    horse[leg1]['axisZ'] = -1
    horse[leg1]['height'] = leglength
    horse[leg1]['radius'] = bellrand(2, 3.5)
    horse[leg1]['texture'] = bellrand(170, 180)

    horse[leg2] = {}
    horse[leg2]['_type'] = 'leg2'
    horse[leg2]['locationX'] = horse[torso]['height'] / 2
    horse[leg2]['locationY'] = -horse[torso]['radius']
    horse[leg2]['locationZ'] = -horse[torso]['radius'] - leglength/2
    horse[leg2]['axisX'] = 0
    horse[leg2]['axisY'] = 0
    horse[leg2]['axisZ'] = -1
    horse[leg2]['height'] = leglength
    horse[leg2]['radius'] = bellrand(2, 3.5)
    horse[leg2]['texture'] = bellrand(170, 180)

    horse[leg3] = {}
    horse[leg3]['_type'] = 'leg3'
    horse[leg3]['locationX'] = -horse[torso]['height'] / 2
    horse[leg3]['locationY'] = horse[torso]['radius']
    horse[leg3]['locationZ'] = -horse[torso]['radius'] - leglength/2
    horse[leg3]['axisX'] = 0
    horse[leg3]['axisY'] = 0
    horse[leg3]['axisZ'] = -1
    horse[leg3]['height'] = leglength
    horse[leg3]['radius'] = bellrand(2, 3.5)
    horse[leg3]['texture'] = bellrand(170, 180)

    horse[leg4] = {}
    horse[leg4]['_type'] = 'leg4'
    horse[leg4]['locationX'] = -horse[torso]['height'] / 2
    horse[leg4]['locationY'] = -horse[torso]['radius']
    horse[leg4]['locationZ'] = -horse[torso]['radius'] - leglength/2
    horse[leg4]['axisX'] = bellrand(0.0, 0.5)
    horse[leg4]['axisY'] = 0
    horse[leg4]['axisZ'] = -1
    horse[leg4]['height'] = leglength
    horse[leg4]['radius'] = bellrand(2, 3.5)
    horse[leg4]['texture'] = bellrand(170, 180)

    neck_height = bellrand(12, 16)

    horse[neck] = {}
    horse[neck]['_type'] = 'neck'
    horse[neck]['locationX'] = (horse[torso]['height'] / 2 + 
                                neck_height / 2.828)
    horse[neck]['locationY'] = 0
    horse[neck]['locationZ'] = (horse[torso]['radius'] + 
                                neck_height / 2.828)
    horse[neck]['axisX'] = 1
    horse[neck]['axisY'] = 0
    horse[neck]['axisZ'] = 1
    horse[neck]['height'] = neck_height
    horse[neck]['radius'] = bellrand(5, 7)
    horse[neck]['texture'] = bellrand(170, 180)

    horse[head] = {}
    horse[head]['_type'] = 'head'
    horse[head]['locationX'] = horse[torso]['height'] / 2 + neck_height / 1.414
    horse[head]['locationY'] = 0
    horse[head]['locationZ'] = horse[torso]['radius'] + neck_height / 1.414
    horse[head]['axisX'] = 1
    horse[head]['axisY'] = bellrand(-1, 1)
    horse[head]['axisZ'] = 0
    horse[head]['height'] = bellrand(18, 22)
    horse[head]['radius'] = bellrand(4, 6)
    horse[head]['texture'] = bellrand(170, 180)

    tail_height = bellrand(26, 33)
    horse[tail] = {}
    horse[tail]['_type'] = 'tail'
    horse[tail]['locationX'] = -horse[torso]['height'] + tail_height
    horse[tail]['locationY'] = 0
    horse[tail]['locationZ'] = 0
    horse[tail]['axisX'] = -1
    horse[tail]['axisY'] = bellrand(-1, 1)
    horse[tail]['axisZ'] = bellrand(-1, 0)
    horse[tail]['height'] = tail_height
    horse[tail]['radius'] = bellrand(1, 2)
    horse[tail]['texture'] = bellrand(170, 180)

    return horse

def generate_giraffe():
    giraffe = {}
    giraffe['_type'] = 'giraffe'

    torso = 'torso'
    leg1 = 'leg1'
    leg2 = 'leg2'
    leg3 = 'leg3'
    leg4 = 'leg4'
    neck = 'neck'
    head = 'head'
    tail = 'tail'

    giraffe[torso] = {}
    giraffe[torso]['_type'] = 'torso'
    giraffe[torso]['locationX'] = 0
    giraffe[torso]['locationY'] = 0
    giraffe[torso]['locationZ'] = 0
    giraffe[torso]['axisX'] = 1
    giraffe[torso]['axisY'] = 0
    giraffe[torso]['axisZ'] = 0
    giraffe[torso]['height'] = bellrand(60, 72)
    giraffe[torso]['radius'] = bellrand(12.5, 17)
    giraffe[torso]['texture'] = bellrand(170, 180)

    leglength = bellrand(58, 70)

    giraffe[leg1] = {}
    giraffe[leg1]['_type'] = 'leg1'
    giraffe[leg1]['locationX'] = giraffe[torso]['height'] / 2
    giraffe[leg1]['locationY'] = giraffe[torso]['radius']
    giraffe[leg1]['locationZ'] = -giraffe[torso]['radius'] - leglength/2
    giraffe[leg1]['axisX'] = bellrand(0, 0.5)
    giraffe[leg1]['axisY'] = 0
    giraffe[leg1]['axisZ'] = -1
    giraffe[leg1]['height'] = leglength
    giraffe[leg1]['radius'] = bellrand(2, 4)
    giraffe[leg1]['texture'] = bellrand(170, 180)

    giraffe[leg2] = {}
    giraffe[leg2]['_type'] = 'leg2'
    giraffe[leg2]['locationX'] = giraffe[torso]['height'] / 2
    giraffe[leg2]['locationY'] = -giraffe[torso]['radius']
    giraffe[leg2]['locationZ'] = -giraffe[torso]['radius'] - leglength/2
    giraffe[leg2]['axisX'] = 0
    giraffe[leg2]['axisY'] = 0
    giraffe[leg2]['axisZ'] = -1
    giraffe[leg2]['height'] = leglength
    giraffe[leg2]['radius'] = bellrand(2, 4)
    giraffe[leg2]['texture'] = bellrand(170, 180)

    giraffe[leg3] = {}
    giraffe[leg3]['_type'] = 'leg3'
    giraffe[leg3]['locationX'] = -giraffe[torso]['height'] / 2
    giraffe[leg3]['locationY'] = giraffe[torso]['radius']
    giraffe[leg3]['locationZ'] = -giraffe[torso]['radius'] - leglength/2
    giraffe[leg3]['axisX'] = 0
    giraffe[leg3]['axisY'] = 0
    giraffe[leg3]['axisZ'] = -1
    giraffe[leg3]['height'] = leglength
    giraffe[leg3]['radius'] = bellrand(2, 4)
    giraffe[leg3]['texture'] = bellrand(170, 180)

    giraffe[leg4] = {}
    giraffe[leg4]['_type'] = 'leg4'
    giraffe[leg4]['locationX'] = -giraffe[torso]['height'] / 2
    giraffe[leg4]['locationY'] = -giraffe[torso]['radius']
    giraffe[leg4]['locationZ'] = -giraffe[torso]['radius'] - leglength/2
    giraffe[leg4]['axisX'] = bellrand(0.0, 0.5)
    giraffe[leg4]['axisY'] = 0
    giraffe[leg4]['axisZ'] = -1
    giraffe[leg4]['height'] = leglength
    giraffe[leg4]['radius'] = bellrand(2, 4)
    giraffe[leg4]['texture'] = bellrand(170, 180)

    neck_height = bellrand(45, 55)

    giraffe[neck] = {}
    giraffe[neck]['_type'] = 'neck'
    giraffe[neck]['locationX'] = (giraffe[torso]['height'] / 2 + 
                                  neck_height / 2.828)
    giraffe[neck]['locationY'] = 0
    giraffe[neck]['locationZ'] = (giraffe[torso]['radius'] + 
                                  neck_height / 2.828)
    giraffe[neck]['axisX'] = 1
    giraffe[neck]['axisY'] = 0
    giraffe[neck]['axisZ'] = 1
    giraffe[neck]['height'] = neck_height
    giraffe[neck]['radius'] = bellrand(5, 9)
    giraffe[neck]['texture'] = bellrand(170, 180)

    giraffe[head] = {}
    giraffe[head]['_type'] = 'head'
    giraffe[head]['locationX'] = (giraffe[torso]['height'] / 2 + 
                                  neck_height / 1.414)
    giraffe[head]['locationY'] = 0
    giraffe[head]['locationZ'] = (giraffe[torso]['radius'] + 
                                  neck_height / 1.414)
    giraffe[head]['axisX'] = 1
    giraffe[head]['axisY'] = bellrand(-1, 1)
    giraffe[head]['axisZ'] = 0
    giraffe[head]['height'] = bellrand(18, 22)
    giraffe[head]['radius'] = bellrand(3.5, 5.5)
    giraffe[head]['texture'] = bellrand(170, 180)

    tail_height = bellrand(20, 25)
    giraffe[tail] = {}
    giraffe[tail]['_type'] = 'tail'
    giraffe[tail]['locationX'] = -giraffe[torso]['height'] + tail_height
    giraffe[tail]['locationY'] = 0
    giraffe[tail]['locationZ'] = 0
    giraffe[tail]['axisX'] = -1
    giraffe[tail]['axisY'] = bellrand(-1, 1)
    giraffe[tail]['axisZ'] = bellrand(-1, 0)
    giraffe[tail]['height'] = tail_height
    giraffe[tail]['radius'] = bellrand(0.5, 1.0)
    giraffe[tail]['texture'] = bellrand(170, 180)

    return giraffe 

