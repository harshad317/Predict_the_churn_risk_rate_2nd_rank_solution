import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from feature_engine.encoding import MeanEncoder
from sklearn.metrics import f1_score
import optuna
import pickle

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

ids = test.customer_id
train.churn_risk_score = train.churn_risk_score.replace(-1, 1)


def preprocessing(df):
    df.gender = df.gender.replace('Unknown', np.nan)
    df.joined_through_referral = df.joined_through_referral.replace('?', np.nan)
    df.medium_of_operation = df.medium_of_operation.replace('?', np.nan)

    df.region_category.fillna('Village', axis=0, inplace=True)
    df.preferred_offer_types.fillna(df.preferred_offer_types.mode()[0], axis=0, inplace=True)
    df.points_in_wallet.fillna(value=df.points_in_wallet.median(), axis=0, inplace=True)
    df.gender.fillna(df.gender.mode()[0], axis=0, inplace=True)
    df.joined_through_referral.fillna(df.joined_through_referral.mode()[0], axis=0, inplace=True)
    df.medium_of_operation.fillna('Both', axis=0, inplace=True)

    df['avg_frequency_login_days'] = df['avg_frequency_login_days'].apply(lambda x: 0 if x == 'Error' else x)
    df['avg_frequency_login_days'] = pd.to_numeric(df['avg_frequency_login_days'])
    df['avg_frequency_login_days'] = df['avg_frequency_login_days'].apply(lambda x: 0 if x < 0 else x)
    df['avg_transaction_value'] = df['avg_transaction_value'].apply(lambda x: 0 if x < 0 else x)
    df['days_since_last_login'] = df['days_since_last_login'].apply(lambda x: 0 if x < 0 else x)
    df['avg_frequency_login_days'] = df['avg_frequency_login_days'].apply(lambda x: 0 if x < 0 else x)
    df['membership_by_refer_id_min'] = df.groupby('referral_id')['membership_category'].transform('min')
    df['membership_by_refer_id_max'] = df.groupby('referral_id')['membership_category'].transform('max')
    return df


def feature_engg(df):
    df['joining_date'] = pd.to_datetime(df['joining_date'], format='%Y-%m-%d')
    df['day'] = df['joining_date'].dt.day
    df['year'] = df['joining_date'].dt.year
    df['month'] = df['joining_date'].dt.month
    df = df.drop('joining_date', axis=1)

    df['last_visit_time'] = pd.to_datetime(df['last_visit_time'], format='%H:%M:%S')
    df['visiting_hour'] = df['last_visit_time'].dt.hour
    df['visiting_min'] = df['last_visit_time'].dt.minute
    df['visiting_sec'] = df['last_visit_time'].dt.second
    df = df.drop('last_visit_time', axis=1)

    df['first_name'] = df.Name.str.split(' ', expand=True)[0]
    naming = df.first_name.value_counts(normalize=True).to_dict()
    df.first_name = df.first_name.map(naming)
    df['hours_since_last_login'] = df.days_since_last_login * 24
    df['actual_time_spent'] = df.avg_time_spent - df.visiting_sec

    df['customer_by_membership'] = df.groupby('membership_category')['customer_id'].transform('count')
    df['points_by_Name'] = df.groupby('points_in_wallet')['Name'].transform('count')
    df['avg_trans_by_security_no'] = df.groupby('avg_frequency_login_days')['security_no'].transform('count')

    df['trans_points_add'] = df['avg_transaction_value'] + df['points_in_wallet']
    df['trans_points_sub'] = df['avg_transaction_value'] - df['points_in_wallet']
    return df


cat_cols = ['gender','region_category','membership_category','joined_through_referral','preferred_offer_types','medium_of_operation','internet_option',
            'used_special_discount','offer_application_preference','complaint_status','feedback', 'membership_by_refer_id_min',
            'membership_by_refer_id_max']

cont_cols = ['age','days_since_last_login','avg_time_spent','avg_transaction_value','avg_frequency_login_days','points_in_wallet','day',
             'year','month','visiting_hour','visiting_min','visiting_sec','first_name','hours_since_last_login','actual_time_spent',
             'customer_by_membership','points_by_Name','avg_trans_by_security_no','trans_points_add','trans_points_sub']


def models(estimators):
    model= XGBClassifier(n_estimators=estimators)
    model.fit(train[xgb_features], target)
    preds = model.predict(test[xgb_features])
    return preds, model

train = preprocessing(train)
test = preprocessing(test)

train = feature_engg(train)
test = feature_engg(test)

target = train.churn_risk_score
xgb_cat_features = []

mean_enc = MeanEncoder(variables= cat_cols)
mean_enc.fit(train[cat_cols], target)
train[cat_cols] = mean_enc.transform(train[cat_cols])
test[cat_cols] = mean_enc.transform(test[cat_cols])
xgb_cat_features.extend(cat_cols)

xgb_train_preds = np.zeros(len(train.index),)
xgb_test_preds = np.zeros(len(test.index),)
xgb_features = xgb_cat_features + cont_cols



preds, model = models(estimators=575)
filename = 'XGBM_model'
pickle.dump(model, open(filename, 'wb'))

submission = pd.DataFrame()
submission['customer_id'] = ids
submission['churn_risk_score'] = preds
submission.to_csv('prediction.csv', index= False)