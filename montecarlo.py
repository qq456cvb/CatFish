import random
from typing import List
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.signal import convolve


def playgame(a, b):
    commons = []
    def playturn(cards: List):
        nonlocal commons
        c = random.choice(cards)
        turn = 1
        while c in commons:
            loc = commons.index(c)
            cards.extend(commons[loc:])
            del commons[loc:]
            turn += 1
            c = random.choice(cards)
        cards.remove(c)
        commons.append(c)
        return turn
    
    cnt = 0
    while True:
        cnt += playturn(a)
        if len(a) == 0:
          break
        cnt += playturn(b)
        if len(b) == 0:
            break
    return cnt


def game_simp(a, b):
    cnt = 0
    while True:
        cnt += 1
        num = np.random.rand()
        if num < 0.5:
            if num < 0.25:
                c = np.random.choice(b)
                a.append(c)
                b.remove(c)
            else:
                c = np.random.choice(a)
                b.append(c)
                a.remove(c)
        if len(a) == 0 or len(b) == 0:
            break
        
    return cnt


def calc_prob_in_range(prob, a, b):
    prob_one_dir = prob[len(prob) // 2:]
    if a > len(prob_one_dir) - 1:
        return 0
    else:
        return np.sum(prob_one_dir[a:b])


# length of trajectory that larger than n, and hit either -c or c
def calc_P(n, c, basic_prob=np.array([0.25, 0.5, 0.25])):
    prob = basic_prob
    for _ in range(n - 1):
        prob = convolve(prob, basic_prob)
    
    if n <= c:
        prob_out = 0
    else:
        prob_out = np.sum(prob[len(prob) // 2 + 1 + c:]) * 2
    
    res = 1. - prob_out
    sign = 1
    for i in range(1, n + 1, 2):
        res -= 2 * sign * calc_prob_in_range(prob, i * c, (i + 2) * c)
        sign *= -1
    return res
    
    
if __name__ == '__main__':
    # cards = list(range(13)) * 4 + [14, 15]
    cards = [1, 1, 2, 2, 3, 3, 4, 4]
    bins = np.zeros((3000,))
    max_cnt = 0
    mean_cnt = 0
    num_trials = 100000
    for _ in tqdm(range(num_trials)):
        random.shuffle(cards)
        cnt = game_simp(cards[:len(cards) // 2], cards[len(cards) // 2:])
        bins[min(cnt, 2999)] += 1
        max_cnt = max(cnt, max_cnt)
        mean_cnt += cnt
    
    mean_cnt /= num_trials
    print(max_cnt, mean_cnt)

    bins = bins[:max_cnt + 1] / num_trials
    print(bins[:10])
    plt.figure()
    plt.plot(np.arange(len(bins)), bins)
    
    c = len(cards) // 2
    P = []
    for n in range(max_cnt + 1):
        P.append(calc_P(n, c))
    
    p = [0]
    for n in range(1, max_cnt + 1):
        p.append(P[n - 1] - P[n])
    p = np.stack(p)
    print(p[:10])    
        
    plt.figure()
    scale = np.max(bins) / np.max(p)
    plt.plot(np.arange(len(p)), bins, '-b', label='Monte-Carlo')
    plt.plot(np.arange(len(p)), p * scale, '-r', label='Analytic')
    plt.legend()
    plt.show()
    