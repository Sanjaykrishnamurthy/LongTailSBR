import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t



def sample(user_train, usernum, itemnum, batch_size, maxlen):

    user = np.random.randint(1, usernum + 1)
    while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

    seq = np.zeros([maxlen], dtype=np.int32)
    pos = np.zeros([maxlen], dtype=np.int32)
    neg = np.zeros([maxlen], dtype=np.int32)
    nxt = user_train[user][-1]
    idx = maxlen - 1

    ts = set(user_train[user])
    for i in reversed(user_train[user][:-1]):
        seq[idx] = i
        pos[idx] = nxt
        if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
        nxt = i
        idx -= 1
        if idx == -1: break

    return (user, seq, pos, neg)

 
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args, head_items, tail_items):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    recommended_items = set()
    
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    
    user_head_coverage = []
    user_tail_coverage = []
    
    item_count = {key: 0 for key in range(itemnum+1)}
    
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        item_idx = torch.tensor(item_idx)
        top_k_indices = predictions.argsort(descending=True) 
        top_k_items = item_idx[top_k_indices]
        # no_popular = np.intersect1d(popular_items, top_k_items.numpy())
        user_rec = set(top_k_items[:20].numpy())
        recommended_items.update(user_rec)
        
        # Item diversity; how the recommended item is diverse between popular/head and unpopular/tail items.
        user_rec_tail = user_rec - head_items
        user_rec_head = user_rec - tail_items
        user_head_coverage.append(len(user_rec_tail)/20)
        user_tail_coverage.append(len(user_rec_head)/20)
        
        # Weighted long tail-coverage.
        for k in user_rec_tail:
            item_count[k]+=1
        
        
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    
    # Item novelity
    user_head_coverage = np.mean(user_head_coverage)
    user_tail_coverage = np.mean(user_tail_coverage)
    
    # Weighted Item-Coverage
    item_user_count = 0
    for k in tail_items:
        item_user_count+= item_count[k]
    weighted_item_coverage = item_user_count/len(tail_items)
    
    
    print(f'user_tail_coverage = {user_tail_coverage:0.3f}, user_head_coverage = {user_head_coverage:0.3f}, \
          weighted_item_coverage= {weighted_item_coverage}')
    return NDCG / valid_user, HT / valid_user, recommended_items, user_head_coverage, user_tail_coverage#, weighted_item_coverage
