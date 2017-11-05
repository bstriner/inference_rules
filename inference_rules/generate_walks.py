import pickle

import numpy as np
from tqdm import tqdm

from .util import make_path
from .util import split_data
from .walker import relations_on_walks, triple_map, calc_reachable_all, calc_answers


def generate_type_lists(facts, r_k):
    """ Return a list of sets by type"""
    type_list = [set() for _ in range(r_k)]
    for i in range(facts.shape[0]):
        r = facts[i, 1]
        t = facts[i, 2]
        type_list[r].add(t)
    return type_list


from .walker import augment


def walks_from_negative_set(train_map, s, valid_negative_set, max_negative_samples, max_depth):
    neg_walks = []
    setsize = len(valid_negative_set)
    if setsize > 0:
        # select up to max_negative_samples
        a = np.arange(setsize)
        n = min(setsize, max_negative_samples)
        idx = np.random.choice(a=a, size=(n,), replace=False)
        neg_ts = [valid_negative_set[i] for i in idx]
        # collect walks
        for neg_t in neg_ts:
            neg_rels = []
            for d in range(max_depth):
                neg_rel = relations_on_walks(tm=train_map, depth=d + 1, s=s, t=neg_t)
                neg_rels.append(neg_rel)
            neg_walks.append({'rels': neg_rels, 'neg_t': neg_t})
    return neg_walks


def generate_walks(facts,
                   holdouts,
                   max_depth,
                   r_k,
                   max_negative_samples,
                   desc='generating walks',
                   all_facts=None):
    all_samples = []

    train_map = triple_map(augment(facts, r_k))
    all_map = None if all_facts is None else triple_map(all_facts)
    type_list = generate_type_lists(facts, r_k) if all_facts is None else generate_type_lists(all_facts, r_k)
    unreachable = 0
    trivial = 0
    candidate_sampled = 0
    supplemented = 0
    undersampled = 0

    it = tqdm(range(holdouts.shape[0]), desc=desc)
    for i in it:
        it.desc = "{} Samples {} ({} full + {} supp + {} under) Unreachable {} Trivial {}".format(desc,
                                                                                                  len(all_samples),
                                                                                                  candidate_sampled,
                                                                                                  supplemented,
                                                                                                  undersampled,
                                                                                                  unreachable,
                                                                                                  trivial)

        s = holdouts[i, 0]
        r = holdouts[i, 1]
        t = holdouts[i, 2]

        # Check if reachable using training data and list walks between s and t
        reachable = False
        walks = []
        for d in range(max_depth):
            walk = relations_on_walks(tm=train_map, depth=d + 1, s=s, t=t)  # (walks, d)
            walks.append(walk)
            if walk is not None:
                reachable = True

        if reachable:
            # candidates and correct answers based on train data or all data
            candidate_set = type_list[r]
            if all_map is None:
                correct_set = calc_answers(tm=train_map, s=s, r=r)
            else:
                correct_set = calc_answers(tm=all_map, s=s, r=r)
            correct_set.add(t)
            # consider only reachable using train data
            reachable_set = calc_reachable_all(tm=train_map, depth=max_depth, s=s)
            # Prefer entities that are reachable candidates but not correct
            valid_negative_set = [e for e in reachable_set if (e not in correct_set) and (e in candidate_set)]
            neg_walks = walks_from_negative_set(train_map=train_map,
                                                s=s,
                                                valid_negative_set=valid_negative_set,
                                                max_negative_samples=max_negative_samples,
                                                max_depth=max_depth)
            # Settle for anything reachable that is not correct
            if len(neg_walks) < max_negative_samples:
                valid_negative_set = [e for e in reachable_set if (e not in correct_set)]
                addl_walks = walks_from_negative_set(train_map=train_map,
                                                     s=s,
                                                     valid_negative_set=valid_negative_set,
                                                     max_negative_samples=max_negative_samples - len(neg_walks),
                                                     max_depth=max_depth)
                neg_walks.extend(addl_walks)
                is_supplemented = True
            else:
                is_supplemented = False
            if len(neg_walks) > 0:
                if len(neg_walks) < max_negative_samples:
                    # not completely full on negative samples
                    undersampled += 1
                elif is_supplemented:
                    supplemented += 1
                else:
                    candidate_sampled += 1

                # append positive and negative sample
                sample = {'s': s, 'r': r, 't': t, 'rels': walks, 'neg_walks': neg_walks}
                all_samples.append(sample)
            else:
                # all reachable answers are correct
                trivial += 1
        else:
            # cannot reach correct answer
            unreachable += 1
    return all_samples, unreachable, trivial, candidate_sampled, supplemented, undersampled


def generate_validation_data(output_path, train, valid, r_k, max_depth, all_facts, max_negative_samples=64):
    make_path(output_path)
    walks, unreachable, trivial, candidate_sampled, supplemented, undersampled = generate_walks(
        facts=train,
        holdouts=valid,
        max_depth=max_depth,
        all_facts=all_facts,
        r_k=r_k,
        max_negative_samples=max_negative_samples,
        desc='Validation Walks')
    with open(output_path, 'wb') as f:
        pickle.dump(walks, f)
    with open(output_path + '.txt', 'w') as f:
        f.write("len(walks): {}\n".format(len(walks)))
        f.write("unreachable: {}\n".format(unreachable))
        f.write("trivial: {}\n".format(trivial))
        f.write("candidate_sampled: {}\n".format(candidate_sampled))
        f.write("supplemented: {}\n".format(supplemented))
        f.write("undersampled: {}\n".format(undersampled))


def generate_training_data(output_path, train, r_k, max_depth, splits=10, max_negative_samples=64):
    make_path(output_path)
    print type(train)
    train_holdout = list(split_data(train, splits))
    train_facts = list(np.concatenate(list(train_holdout[j] for j in range(splits) if j != i), axis=0)
                       for i in range(splits))

    all_walks = []
    unreachable_tot = 0
    trivial_tot = 0
    candidate_sampled_tot = 0
    supplemented_tot = 0
    undersampled_tot = 0
    for i in tqdm(range(splits), desc='Training data'):
        walks, unreachable, trivial, candidate_sampled, supplemented, undersampled = generate_walks(
            facts=train_facts[i],
            holdouts=train_holdout[i],
            max_depth=max_depth,
            r_k=r_k,
            max_negative_samples=max_negative_samples,
            desc='Split {}'.format(i))
        all_walks.extend(walks)
        unreachable_tot += unreachable
        trivial_tot += trivial
        candidate_sampled_tot += candidate_sampled
        supplemented_tot += supplemented
        undersampled_tot += undersampled

    with open(output_path, 'wb') as f:
        pickle.dump(all_walks, f)
    with open(output_path + '.txt', 'w') as f:
        f.write("len(walks): {}\n".format(len(all_walks)))
        f.write("unreachable: {}\n".format(unreachable_tot))
        f.write("trivial: {}\n".format(trivial_tot))
        f.write("candidate_sampled: {}\n".format(candidate_sampled_tot))
        f.write("supplemented: {}\n".format(supplemented_tot))
        f.write("undersampled: {}\n".format(undersampled_tot))


def generate_training_walks(output_path, facts, holdout, r_k, max_depth, max_negative_samples=64):
    make_path(output_path)
    walks, unreachable, trivial, candidate_sampled, supplemented, undersampled = generate_walks(
        facts=facts,
        holdouts=holdout,
        max_depth=max_depth,
        r_k=r_k,
        max_negative_samples=max_negative_samples,
        desc='Training Data')

    with open(output_path, 'wb') as f:
        pickle.dump(walks, f)
    with open(output_path + '.txt', 'w') as f:
        f.write("len(walks): {}\n".format(len(walks)))
        f.write("unreachable: {}\n".format(unreachable))
        f.write("trivial: {}\n".format(trivial))
        f.write("candidate_sampled: {}\n".format(candidate_sampled))
        f.write("supplemented: {}\n".format(supplemented))
        f.write("undersampled: {}\n".format(undersampled))
    with open(output_path + '.pickle', 'wb') as f:
        data = {'walks': len(walks),
                'unreachable': unreachable,
                'trivial': trivial,
                'candidate_sampled': candidate_sampled,
                'supplemented': supplemented,
                'undersampled': undersampled
                }
        pickle.dump(data, f)
