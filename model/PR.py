

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

def get_polynomial_regression(n_degree, alpha):
    pipe = Pipeline([('PolynomialFeatures', PolynomialFeatures(n_degree)), ('Ridge', Ridge(alpha= alpha))])
    return pipe