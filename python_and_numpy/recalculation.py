import numpy as np


def recalculate_data(data, clusters):
    """
    :param np_data: данные об изменениях цены N акций (axis 0) в M периодах (axis 1) - размерность N x M
    :param np_clusters: метки кластеров, размерностью 1 x N. Всего кластеров - K
    :return: пересчитанные данные в среднем для акций по кластерам, размерность: K x M
    """
    k = np.unique(clusters).max()
    n, m = data.shape
    res = np.ones((k, m))
    clusters_sizes = np.zeros(k, dtype=np.int64)
    clusters = clusters.astype(np.int64)

    for i in range(n):
        clusters_sizes[clusters[i] - 1] += 1
        for j in range(m):
            res[clusters[i] - 1][j] *= data[i][j]

    for i in range(k):
        for j in range(m):
            res[i][j] = res[i][j] ** (1 / clusters_sizes[i])
    return res


def recalculate_returns(returns, clusters):
    """
    :param np_returns: данные о прибыльности N акций - размерность (N,)
    :param np_clusters: метки кластеров, размерностью 1 x N. Всего кластеров - K
    :return: пересчитанные данные в среднем для акций по кластерам, размерность: K
    """
    k = np.unique(clusters).max()
    n = returns.shape[0]
    clusters = clusters.astype(np.int64)
    res = np.ones((k,))
    clusters_sizes = np.zeros(k, dtype=np.int64)

    for i in range(n):
        clusters_sizes[clusters[i] - 1] += 1
        res[clusters[i] - 1] *= returns[i]

    for i in range(k):
        res[i] = res[i] ** (1 / clusters_sizes[i])
    return np.array(res)
