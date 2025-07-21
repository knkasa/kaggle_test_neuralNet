print('starting...')

import os
import sys
import gc
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from autogluon.tabular import TabularPredictor
import featuretools as ft
import boto3

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

# df_train(1200000) df_test(800000)
train_rows = 1_200_000
test_rows = 800_000

target = 'Premium Amount'
use_featuretools = True
use_pca = True
seed = int(os.getenv('SEED'))
num_neighbors = 50

print(f'SEED:{seed}')

'''
s3 = boto3.client('s3')
bucket = 'test-ecs-s3'

print('downloading data...')
key = 'kaggle_input/train.csv'
local_file = '/tmp/train.csv'
s3.download_file(bucket, key, local_file)
df_train = pd.read_csv('/tmp/train.csv', parse_dates=['Policy Start Date'])
df_train = df_train.sample(frac=1, random_state=seed)

key = 'kaggle_input/test.csv'
local_file = '/tmp/test.csv'
s3.download_file(bucket, key, local_file)
df_test = pd.read_csv('/tmp/test.csv', parse_dates=['Policy Start Date'])

df = pd.concat([df_train, df_test], axis=0)
df.reset_index(drop=True, inplace=True)

#print(df.isnull().sum()/len(df))

numeric_cols = df.select_dtypes(include=[np.number]).columns[1:-1] # ignore id, target columns
categorical_cols = df.select_dtypes(include=['object']).columns

df['NumDays'] = (df['Policy Start Date'].max() - df['Policy Start Date']).dt.days
df.drop(columns=['id', 'Policy Start Date'], inplace=True)
numeric_cols = [*numeric_cols, 'NumDays']

df = pd.get_dummies(df, drop_first=True)

scalerx = StandardScaler()
#scalerx = PowerTransformer(method='yeo-johnson', standardize=True)
df[numeric_cols] = scalerx.fit_transform(df[numeric_cols])

for col in df.select_dtypes(include='bool').columns:
    df[col] = df[col].astype(int)

input_cols = df.columns.difference([target])

print('imputing...')
imputer = KNNImputer(n_neighbors=num_neighbors)
df_impute1 = pd.DataFrame(imputer.fit_transform(df.loc[:100_000]), columns=df.columns, index=df.loc[:100_000].index)
df_impute2 = pd.DataFrame(imputer.transform(df.loc[100_000:]), columns=df.columns, index=df.loc[100_000:].index)
df = pd.concat([df_impute1, df_impute2], axis=0)

print('uploading imputed data...')
local_file = '/tmp/train_imputed.parquet'
df.to_parquet(local_file)
key = 'kaggle_input/train_imputed.parquet'
s3.upload_file(local_file, bucket, key)
#exit()
'''

print('downloading ...')
s3 = boto3.client('s3')
bucket = 'test-ecs-s3'
key = 'kaggle_input/train_imputed.parquet'
local_file = '/tmp/train_imputed.parquet'
s3.download_file(bucket, key, local_file)

df = pd.read_parquet(local_file)
input_cols = df.columns.difference([target])
print('data loaded')

if use_featuretools:
    df['id'] = range(len(df))
    es = ft.EntitySet(id="new_data")
    es = es.add_dataframe(
        dataframe_name="df",
        dataframe=df[input_cols],
        index="id"
        )

    transformation_primitives = [
        "add_numeric",           # col1 + col2
        #"subtract_numeric",      # col1 - col2  
        "multiply_numeric",      # col1 * col2
        #"divide_numeric",        # col1 / col2
        #"square_root",           # sqrt(col1)
        #"square",                # col1^2
        "natural_logarithm",     # ln(col1)
        #"absolute",              # abs(col1)
        "percentile",            # percentile rank of values
        #"cum_sum",               # cumulative sum
        #"cum_mean",              # cumulative mean
        #"cum_max",               # cumulative max
        #"cum_min"                # cumulative min
        ]

    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="df",
        trans_primitives=transformation_primitives,
        max_depth=1,
        verbose=False
        )

    feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)
    feature_matrix = feature_matrix.fillna(feature_matrix.median())
    print(f"num aggregated cols:{feature_matrix.shape[1]}")
    cols_list = feature_matrix.columns
    feature_matrix[target] = df[target].values
    df = feature_matrix.copy()
    del feature_matrix
    gc.collect()

pca = PCA()
pca.fit(df[input_cols])
cum_pca_val = np.cumsum(pca.explained_variance_ratio_)
num_pca_cols = np.where(cum_pca_val>=0.95)[0][0] + 1
print(f"number of pca cols:{num_pca_cols}")

if use_pca:
    print("Applying PCA...")
    pca_final = PCA(n_components=num_pca_cols)
    cols_list = [f'col{n+1}' for n in range(num_pca_cols)]
    df_pca = pd.DataFrame(pca_final.fit_transform(df[input_cols]), columns=cols_list, index=df.index)
    df_pca.loc[:, target] = df[target]

    df_train = df_pca.iloc[:int(train_rows*0.8)].copy()
    df_val = df_pca.iloc[int(train_rows*0.8):-test_rows].copy()
    df_test = df_pca.iloc[-test_rows:].copy()
    input_dim = num_pca_cols
else:
    df_train = df.iloc[:int(train_rows*0.8)].copy()
    df_val = df.iloc[int(train_rows*0.8):-test_rows].copy()
    df_test = df.iloc[-test_rows:].copy()
    input_dim = df.shape[1]-1
    if not use_featuretools:
        cols_list = input_cols

del df
gc.collect()

#scalery = StandardScaler()
#scalery = PowerTransformer(method='yeo-johnson', standardize=True)
#df_train.loc[:, target] = scalery.fit_transform(df_train[[target]])
#df_test.loc[:, target] = scalery.transform(df_test[[target]])

df_train.loc[:, target] = np.log(df_train[target]+1)
df_val.loc[:, target] = np.log(df_val[target]+1)

# Only takes highest target values
print('only keep high target')
df_train = df_train.sort_values(by=target, ascending=False)[:int(train_rows*0.3)]
df_val = df_val.sort_values(by=target, ascending=False)[:int(train_rows*0.3)]

if input_dim!=len(cols_list): raise Exception("input_din != len(cols_list)")
print(f"input dim:{input_dim}")

print('training starting...')
predictor = TabularPredictor(
    label=target,
    problem_type='regression',
    eval_metric='root_mean_squared_error',  #  'mean_absolute_error', 'r2'
    #path='./maximum_performance_models',
    verbosity=0
    )

predictor.fit(
    df_train,
    time_limit=200000,  
    presets='high_quality',
    #hyperparameters=advanced_hyperparameters,
    num_bag_folds=4,  # each folds creates one model.
    num_bag_sets=1,    # number of bagging model.
    num_stack_levels=2,  # Multi-level stacking
    refit_full=True,   # Refit on full training data
    set_best_to_refit_full=True,
    keep_only_best=False,  # Keep multiple models for analysis
    save_space=False,  # Keep all models for inspection
    #excluded_model_types=['RF', 'XT', 'KNN'],  
    included_model_types=['NN_TORCH'],  
    ag_args_fit={'seed': seed},
    #random_seed=seed,
    )

print("Done fitting.")
leaderboard = predictor.leaderboard(df_val, extra_info=True)

model_info = predictor.info()
best_model_name = model_info['best_model']
top_model_name = leaderboard['model'][0]

print(f"best_model:{best_model_name}  top_model:{top_model_name}")
preds_best = predictor.predict(df_val.drop(target, axis=1), model=best_model_name).values
preds_top = predictor.predict(df_val.drop(target, axis=1), model=top_model_name).values

score_best = np.sqrt(np.mean(np.square(df_val[target] - preds_best)))
score_top = np.sqrt(np.mean(np.square(df_val[target] - preds_top)))
print(f"score_bestNN:{score_best}  score_topNN:{score_top}")

inf_best = predictor.predict(df_test.drop(target, axis=1), model=best_model_name).values
inf_top = predictor.predict(df_test.drop(target, axis=1), model=top_model_name).values

df_test.loc[:, 'preds_best'] = np.exp(inf_best-1)
df_test.loc[:, 'preds_top'] = np.exp(inf_top-1)

print('uploading result.')
local_file = f'/tmp/predsNN_high_{seed}.csv'
df_test.to_csv(local_file, index=None)
key = f'kaggle_output/predsNN_high_{seed}.csv'
s3.upload_file(local_file, bucket, key)

print('uploading leaderboard.')
local_file = f'/tmp/res_autogluonNN_high_{seed}.csv'
leaderboard.to_csv(local_file, index=None)
key = f'kaggle_output/res_autogluonNN_high_{seed}.csv'
s3.upload_file(local_file, bucket, key)

print("All done!!")
