from sklearn.datasets import load_iris
from sklearn.datasets import load_boston

from model_structure.ClassifierModels import MyAdaboost
from model_structure.RegressorModels import MyBR
from model_structure.utils import predict_from_model


def test_clf():

    X, y = load_iris(return_X_y=True, as_frame=True)
    clf = MyAdaboost()
    result = clf.simple_model(X, y, model_save_path='./save_models/')
    print(result)

def test_clf_predict():
    path = './save_models/[2022-06-30 17-14-29] simple_model_Adaboost.model'
    X = [['1', '2', '3', '4']]
    print(predict_from_model(path, X))


def test_reg():
    X, y = load_boston(return_X_y=True)
    reg = MyBR()
    result = reg.simple_model(X, y)
    print(result)


test_clf_predict()