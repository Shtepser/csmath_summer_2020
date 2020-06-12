from timeit import timeit

import numpy as np

from portfolio_optimization import optimize_use_clustering, optimize_port, form_clusters, calc_risk, calc_revenue


def prepare_data(raw_data: np.ndarray) -> np.ndarray:
    """
    Пересчитывает "сырые" данные по акциям в формат: первое значение - 1.0,
        дальше - данные изменений цены по сравнению с первым значением
    :param raw_data: исходные данные, каждая строка массива - одна акция
    :returns: нормализованные данные
    """
    return np.apply_along_axis(lambda x: x / x[0], axis=1, arr=raw_data)


def calc_loss(data, return_value, clust_thresold, clust_dist):
    """
    :param data: данные о ценах акций для кластеризации, формат:  каждая строка массива - \
        данные об изменениях цен за период в % с начала периода (т.е. первое значение = 1.0)
    :param return_value: доходность кластеризованного портфеля
    :param clust_thresold: пороговое значение для кластеризации, если нет - рассчитывается автоматически
    :param clust_dist: пороговое значение кластеризации,
    :return: разница в доходности кластеризованного и некластеризованного портфелей с одинаковым риском,
             снижение размерности (во сколько раз)
    """
    returns = data[:, -1]
    w_clust = optimize_use_clustering(data, return_value, clust_thresold, clust_dist, returns, 'return_bound')
    _, n_of_clusts = form_clusters(data, clust_thresold, clust_dist, return_count=True)
    w_non = optimize_port(data, calc_risk(w_clust, np.cov(data)), returns, 'risk_bound')
    return calc_revenue(w_non, returns) - calc_revenue(w_clust, returns), data.shape[0] / n_of_clusts


def compare_optimizations(data, return_value, clust_thresold, clust_dist, numbers=30):
    returns = data[:, -1]
    loss, economy = calc_loss(data, return_value, clust_thresold, clust_dist)
    clustered_time = timeit(lambda: optimize_use_clustering(data, return_value, clust_thresold, clust_dist, returns),
                            number=numbers) / numbers
    vanilla_time = timeit(lambda: optimize_port(data, return_value, returns),
                          number=numbers) / numbers
    speedup = vanilla_time / clustered_time
    return speedup, economy, loss
