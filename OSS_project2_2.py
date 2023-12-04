import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


def sort_dataset(dataset_df):
    # TODO: Implement this function
    df = dataset_df.sort_values(by='year', ascending=True)
    # dataset_df 를 year 에 대해 정렬, 결과를 가지는 df 생성

    return df


def split_dataset(dataset_df):
    # TODO: Implement this function
    dataset_df['salary'] *= 0.001
    # label 에 0.001 곱해서 rescale

    # train dataframe 과 test dataframe 분리
    train_df = dataset_df[:1718]
    # train dataframe 에 split
    test_df = dataset_df[1718:]
    # test dataframe 에 split

    x_train = train_df.drop(columns=['salary'], axis=1)
    # label 에 사용할 salary 삭제, 독립 변수 x_train dataset 에 저장
    y_train = train_df['salary']
    # 종속 변수 y_train 에 target (salary) dataset 저장

    x_test = test_df.drop(columns=['salary'], axis=1)
    # label 에 사용할 salary 삭제, 독립 변수 x_test dataset 에 저장
    y_test = test_df['salary']
    # 종속 변수 y_test 에 target (salary) dataset 저장

    return x_train, x_test, y_train, y_test


def extract_numerical_cols(dataset_df):
    # TODO: Implement this function
    ret = dataset_df[['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP',
                      'fly', 'war']]
    # numerical_cols 만 가지는 dataset 생성

    return ret


def train_predict_decision_tree(X_train, Y_train, X_test):
    # TODO: Implement this function
    model = DecisionTreeRegressor()
    # DecisionTree model 생성
    model.fit(X_train, Y_train)
    # 모델 학습

    ret = model.predict(X_test)
    # 결과 예측

    return ret


def train_predict_random_forest(X_train, Y_train, X_test):
    # TODO: Implement this function
    model = RandomForestRegressor()
    # RandomForestRegressor model 생성
    model.fit(X_train, Y_train)
    # 모델 학습

    ret = model.predict(X_test)
    # 결과 예측

    return ret


def train_predict_svm(X_train, Y_train, X_test):
    # TODO: Implement this function
    pipeline = make_pipeline(StandardScaler(), SVR())
    # pipeline 생성
    pipeline.fit(X_train, Y_train)
    # 모델 학습

    ret = pipeline.predict(X_test)
    # 결과 예측

    return ret


def calculate_RMSE(labels, predictions):
    # TODO: Implement this function
    ret = np.sqrt(mean_squared_error(labels, predictions))
    # RMSE 계산

    return ret


if __name__ == '__main__':
    # DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
    data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

    sorted_df = sort_dataset(data_df)
    X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)

    X_train = extract_numerical_cols(X_train)
    X_test = extract_numerical_cols(X_test)

    dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
    rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
    svm_predictions = train_predict_svm(X_train, Y_train, X_test)

    print("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))
    print("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))
    print("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))
