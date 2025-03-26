def dot_product(v, w):
    res = 0
    for i in range(len(v)):
        res += v[i] * w[i]
    return res


