import numpy as np

def augment(triples, r_k):
    for t in triples:
        yield t
        yield (t[2], t[1] + r_k, t[0])


def triple_map(triples):
    tm = {}
    for trip in triples:
        s = trip[0]
        if s not in tm:
            tm[s] = []
        tm[s].append(trip)
    return tm


def walk(tm, depth, s, steps):
    # generate walks from s
    if depth > 0:
        for trip in tm.get(s, []):
            for w in walk(tm, depth - 1, trip[2], steps + [trip]):
                yield w
    else:
        yield steps

def calc_reachable(tm, depth, s):
    ents = set()
    for w in walk(tm, depth, s, steps=[]):
        for step in w:
            ents.add(step[2])
    return ents

def calc_reachable_all(tm, depth, s):
    sets = []
    for i in range(depth):
        sets.append(calc_reachable(tm, i+1, s))
    return set().union(*sets)

def count_reachable(tm, depth, s):
    ents = set()
    for w in walk(tm, depth, s, steps=[]):
        for step in w:
            ents.add(step[2])
    return len(ents)


def is_reachable(tm, depth, s, t):
    for w in walk(tm, depth, s, steps=[]):
        if on_path(t, w):
            return True
    return False


def on_path(t, w):
    for step in w:
        if step[0] == t or step[2] == t:
            return True
    return False


def entities_ending_on(tm, depth, s, t):
    ents = set()
    for w in walk(tm, depth, s, steps=[]):
        if w[-1][2] == t:
            for step in w:
                ents.add(step[0])
                ents.add(step[2])
    return ents


def relations_on_walks(tm, depth, s, t):
    rels = []
    for w in walk(tm, depth, s, steps=[]):
        if w[-1][2] == t:
            rs = [s[1] for s in w]
            rels.append(rs)
    if len(rels) > 0:
        return np.array(rels)
    else:
        return None

def calc_answers(tm, s, r):
    triples = tm[s]
    ans = set()
    for trip in triples:
        if trip[1] == r:
            ans.add(trip[2])
    return ans