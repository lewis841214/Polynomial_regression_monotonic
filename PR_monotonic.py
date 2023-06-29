from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import torch
class Poly_linear_regression_grad():
    '''
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Ridge
    from model.Ridge_shift_regression import Ridge_Shift
    from model.model import polynomial_regression
    from model.linear_regression_grad import Poly_linear_regression_grad
    import numpy as np
    scale = 1
    r_scale = 2
    shift = 0.2
    num = 5
    xs = [ele / scale - shift for ele in range(num)]
    ys = [np.exp(-ele ) + np.random.random() / r_scale for ele in xs]

    model = Poly_linear_regression_grad(n_order= 4 ,lr = 0.00001,  momentum = 0.99, epoch= 100000, verbal= False, lambda_ = 0.1)
    model.fit(np.array(xs).reshape(-1, 1), np.array(ys), range_ = [-0.1, 4.2] )
    pred_x = np.arange(-0.1, 4.2, 0.1)
    pred_y = model.predict(pred_x.reshape(-1, 1))
    plt.scatter(xs, ys)
    plt.plot(pred_x, pred_y)
    plt.show()
    '''
    def  __init__(self, n_order, lr = 0.0001, momentum = 0.8, epoch = 1000, verbal = False, lambda_ = 0.01 ):
        """
        Example code
        n_samples, n_features = 3, 2
        """
        self.n_order = n_order
        self.lr = lr
        self.momentum = momentum
        self.epoch = epoch
        self.verbal = verbal
        self.lambda_ = lambda_
    def fit(self, x, y, range_ = None, direction = 'decreasing'):
        '''
        x : numpy : N_len x m_feature, 
        y : numpy : N_len


        '''
        self._range = range_
        self._direction = direction
        self.poly = PolynomialFeatures(degree = self.n_order)
        x_trans = self.poly.fit_transform(x)
        self.n_features = x_trans.shape[1]
        LR =  LinearRegression(fit_intercept=False)
        LR.fit(x_trans, y)

        coef = np.array( LR.coef_.tolist()) / 1000#+ 1
        self.coef_ = torch.from_numpy(coef)
        self.coef_.requires_grad = True
        self.x = torch.from_numpy(x_trans)
        self.y = torch.from_numpy(y)

        # preparing for monotonic loss
        if self._range != None:
            left, right, step = self._range[0], self._range[1], (self._range[1] - self._range[0] ) / 100
            x_range = np.arange(left, right, step).reshape(-1, 1)
            poly__ = PolynomialFeatures(degree = self.n_order - 1)
            self.x_trans__ = poly__.fit_transform(x_range)
            self.x__ = torch.from_numpy(self.x_trans__)

        self.op = torch.optim.SGD([self.coef_], lr=self.lr, momentum = self.momentum)
        self.train(self.epoch)

    def train(self, n_epoch = 1000):
        for _ in range(n_epoch):
            # acc loss 
            predict_y = self.x @ self.coef_.T
            loss1 =  torch.sum((predict_y - self.y) ** 2)
            
            if self._range != None:
                # breakpoint()
                f_derivative = self.x__ @ ( self.coef_[1:] * ( torch.arange(self.n_features - 1 ) + 1 ) )
                if self._direction == 'decreasing':
                    loss2 = torch.sum(f_derivative[f_derivative > 0 ] ** 2)
                else:
                    loss2 = torch.sum(-f_derivative[f_derivative < 0 ]** 2)
                # print('f_derivative', f_derivative)
            else:
                loss2 = 0
            # loss2 = loss1 * 0
            loss = loss1 + self.lambda_ *loss2
            self.op.zero_grad()
            loss.backward()
            # loss1.backward()
            self.op.step()



            if self.verbal and _%10 == 0 :
                print('loss: ', loss.item(), ' loss1: ', loss1.item(), ' loss2: ', loss2.item(), 'self.coef_', self.coef_.detach().numpy())
                # print('predict_y', predict_y)
                # print('self.y', self.y)
            # breakpoint()
        # loss2 = 
        print('final training loss', loss1)
        # print('predict_y', predict_y)
        # print('self.y', self.y)
        # breakpoint()
    def predict(self, x):
        x_trans = self.poly.fit_transform(x)
        pred = x_trans @ self.coef_.detach().numpy()
        return pred

if __name__ == '__main__':
    PLRG = Poly_linear_regression_grad(3)
    rng = np.random.RandomState(0)
    n_samples = 3
    n_features = 1
    y = rng.randn(n_samples)
    X = rng.randn(n_samples, n_features)
    PLRG.fit(X,y)
    print(' PLRG.pred(X)',  PLRG.predict(X))
    breakpoint()
