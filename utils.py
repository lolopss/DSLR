def mean(n):
    return sum(n) / len(n)


def std(n): #ecart type :D
    return (sum([(x - mean(n)) ** 2 for x in n]) / len(n)) ** 0.5

def min(n):
    return sorted(n)[0]

def max(n):
    return sorted(n)[-1]

def median(n):
    n = sorted(n)
    if len(n) % 2 == 0:
        return (n[len(n) // 2] + n[len(n) // 2 - 1]) / 2
    return n[len(n) // 2]


def percentile(n, percentiles):
    if isinstance(percentiles, int):  # If a single integer is passed, return the corresponding percentile value
        percentiles = percentiles / 100
        n = sorted(n)
        index = int(len(n) * percentiles)
        return n[index]
    else:
        percentiles = [p / 100 for p in percentiles]  # Ensure proper scaling

    n = sorted(n)
    index = [int(len(n) * p) for p in percentiles]
    return [n[i] for i in index]


def array(n, dtype=float):
    return [dtype(x) for x in n if x]