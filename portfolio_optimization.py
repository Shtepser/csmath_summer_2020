import numpy as np
import cvxpy as cp
from scipy.cluster.hierarchy import fcluster, linkage

from cythonized.recalculation import recalculate_data, recalculate_returns


def optimize_use_clustering(data, value, t, dist, returns, crit="return_bound"):
    """
    :param data: данные о ценах акций для кластеризации, формат:  каждая строка массива -
        данные об изменениях цен за период в % с начала периода (т.е. первое значение = 1.0)
    :param t: пороговое значение для кластеризации
    :param dist: 'single', 'complete', 'average'
    :param value: уровень критерия - риск для crit="risk_bound", доходность для crit="return_bound",
        gamma для crit="risk_return"
    :param returns: вектор доходностей акций
    :param crit: способ оптимизации - по заданному риску ("risk_bound"), заданной доходности ("return_bound")
        соотношению риска и доходности
    :return: веса каждой акции в портфеле
    """
    clusters = form_clusters(data, t, dist)
    clustered_data = recalculate_data(data, clusters)
    clustered_returns = recalculate_returns(returns, clusters)
    w = optimize_port(clustered_data, value, clustered_returns, crit)
    x = clusters_weights(w, clusters)
    return x


# NOT TESTED
def distance_matrix(data):
    R = np.corrcoef(data)
    if not np.all(np.isfinite(R)):
        R = np.nan_to_num(R, posinf=1.0, neginf=-1.0)
    return np.ones_like(R) - R


def form_clusters(x, t, dist, return_count=False):
    link = linkage(x, metric='correlation', method=dist)
    res = fcluster(Z=link, t=t, criterion='distance')
    if return_count:
        res = res, np.unique(res).max()
    return res


# noinspection DuplicatedCode
def optimize_port(data, value, returns, criterion='return_bound'):
    """
    :param data: данные о ценах акций для кластеризации, формат:  каждая строка массива - \
        данные об изменениях цен за период в % с начала периода (т.е. первое значение = 1.0)
    :param value: уровень критерия - риск для criterion="risk_bound", доходность для criterion="return_bound",\
        gamma для criterion="risk_return"
    :param criterion: способ оптимизации - по заданному риску ("risk_bound"), заданной доходности ("return_bound")\
        соотношению риска и доходности
    :param returns: доходности акций
    :return: веса каждой акции в портфеле
    """
    CV = np.cov(data)
    weights_var = cp.Variable(data.shape[0])
    return_var = returns * weights_var
    risk_var = cp.quad_form(weights_var, CV)
    if criterion == "risk_bound":
        prob = cp.Problem(cp.Maximize(return_var), [risk_var <= cp.Parameter(name="max_risk", value=value, nonneg=True),
                                                    cp.sum(weights_var) == 1, weights_var >= 0])
    elif criterion == "return_bound":
        prob = cp.Problem(cp.Minimize(risk_var), [return_var >= cp.Parameter(name="min_ret", value=value, nonneg=True),
                                                  cp.sum(weights_var) == 1, weights_var >= 0])
    elif criterion == "risk_return":
        prob = cp.Problem(cp.Maximize(return_var - cp.Parameter(name="gamma", value=value, nonneg=True) * risk_var),
                          [cp.sum(weights_var) == 1, weights_var >= 0])
    prob.solve()
    return weights_var.value


def clusters_weights(W, clusters):
    _, clusters_sizes = np.unique(clusters, return_counts=True)
    function = np.vectorize(lambda x: (W / clusters_sizes)[x - 1])
    return function(clusters)


def calc_risk(w: np.ndarray, cv: np.ndarray) -> float:
    """
    :param w: вектор весов акций в портфеле
    :param cv: ковариационная матрица доходностей акций в портфеле
    :return: риск портфеля
    """
    if len(w.shape) == 1:
        w = w.reshape(1, w.shape[0])
    return (w.T.dot(w) * cv).sum()


def calc_revenue(w: np.ndarray, d: np.ndarray) -> float:
    """
    :param w: вектор весов акций в портфеле
    :param d: вектор доходностей акций в портфеле
    :return: ожидаемая доходность портфеля
    """
    return w.dot(d)
