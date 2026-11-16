#!/usr/bin/env python
"""Train the final wine-quality model and save it to model.bin.

Usage:
    python train.py [--data-dir data] [--output model.bin]
"""
import argparse
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split

SEED = 1
GOOD_THRESHOLD = 7  # quality >= 7 -> "good"

NUMERICAL = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
    'ph', 'sulphates', 'alcohol',
]
CATEGORICAL = ['wine_type']
FEATURES = NUMERICAL + CATEGORICAL

# Parameters selected in notebook.ipynb (5-fold CV ROC AUC ~0.848)
RF_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=SEED,
    n_jobs=-1,
)


def load_data(data_dir):
    df_red = pd.read_csv(f'{data_dir}/winequality-red.csv', sep=';')
    df_white = pd.read_csv(f'{data_dir}/winequality-white.csv', sep=';')
    df_red['wine_type'] = 'red'
    df_white['wine_type'] = 'white'

    df = pd.concat([df_red, df_white], ignore_index=True)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df = df.drop_duplicates().reset_index(drop=True)
    df['good'] = (df.quality >= GOOD_THRESHOLD).astype(int)
    return df


def train(df, params=RF_PARAMS):
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(df[FEATURES].to_dict(orient='records'))
    model = RandomForestClassifier(**params)
    model.fit(X, df.good.values)
    return dv, model


def predict(df, dv, model):
    X = dv.transform(df[FEATURES].to_dict(orient='records'))
    return model.predict_proba(X)[:, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output', default='model.bin')
    args = parser.parse_args()

    df = load_data(args.data_dir)
    print(f'Loaded {len(df)} unique wines, good rate = {df.good.mean():.3f}')

    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df.good)
    df_full_train = df_full_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # 5-fold cross-validation on the full training set
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(df_full_train)):
        d_tr, d_va = df_full_train.iloc[tr_idx], df_full_train.iloc[va_idx]
        dv, model = train(d_tr)
        auc = roc_auc_score(d_va.good.values, predict(d_va, dv, model))
        aucs.append(auc)
        print(f'  fold {fold}: AUC = {auc:.4f}')
    print(f'CV AUC = {pd.Series(aucs).mean():.4f} +- {pd.Series(aucs).std(ddof=0):.4f}')

    # Final model on full train, evaluated on the held-out test set
    dv, model = train(df_full_train)
    test_auc = roc_auc_score(df_test.good.values, predict(df_test, dv, model))
    print(f'Test AUC = {test_auc:.4f}')

    # Refit on all data for the production model
    dv, model = train(df)
    with open(args.output, 'wb') as f_out:
        pickle.dump((dv, model), f_out)
    print(f'Model saved to {args.output}')


if __name__ == '__main__':
    main()
