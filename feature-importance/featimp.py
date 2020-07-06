import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_boston
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



def plot_spearman_rank_matrix(df):
    sns.set(style="white")
    corr = df.corr(method='spearman')
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    return corr


def perform_PCA(df, features):

    # Separating out the features
    x = df.loc[:, features].values
    # Separating out the target
    y = df.loc[:, ['target']].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=3)
    fit = pca.fit(x)
    print("explained_variance_ratio:", fit.explained_variance_ratio_)
    print("components:", fit.components_)


def get_mRMR_dict(features, target, corr):
    S = len(features)-1
    imp_dict = {}
    for feature in features:
        row = corr.loc[[feature]]
        self_corr = float(row.loc[:, feature])
        target_corr = float(row.loc[:, target])
        sum_corr_with_other_features = float(row.sum(axis=1))-self_corr-target_corr
        feature_imp = target_corr-((1/float(S))*(sum_corr_with_other_features))
        imp_dict[feature] = feature_imp
    return imp_dict


def dropcol_importances(model, X_train, y_train, X_test, y_test):
    model_ = clone(model)
    model_.random_state = 999
    model_.fit(X_train, y_train)
    baseline = mean_squared_error(y_test, model_.predict(X_test))
    imp = {}
    for col in X_train.columns:
        X = X_train.drop(col, axis=1)
        X_t = X_test.drop(col, axis=1)
        model_ = clone(model)
        model_.random_state = 999
        model_.fit(X, y_train)
        m = mean_squared_error(y_test, model_.predict(X_t))
        imp[col] = baseline - m
    importances = imp
    feature_imp_df = pd.DataFrame(importances.items(), columns=['feature', 'Importance_Score'])
    feature_imp_df['Importance_Score'] = -1 * feature_imp_df['Importance_Score']
    feature_imp_df.loc[(feature_imp_df.Importance_Score > 0), 'Importance'] = 'Positive'
    feature_imp_df.loc[(feature_imp_df.Importance_Score <= 0), 'Importance'] = 'Negative'
    feature_imp_df = feature_imp_df.sort_values(by=['Importance_Score'], ascending=False)
    return feature_imp_df


def permutation_importances(model, X_train, y_train,  X_test, y_test):
    model.fit(X_train, y_train)
    baseline = mean_squared_error(y_test, model.predict(X_test))
    imp = {}
    for col in X_test.columns:
        save = X_test[col].copy()
        X_test[col] = np.random.permutation(X_test[col])
        m = mean_squared_error(y_test, model.predict(X_test))
        X_test[col] = save
        imp[col] = (baseline - m)
    importances_perm = imp
    feature_imp_df_perm = pd.DataFrame(importances_perm.items(), columns=['feature', 'Importance_Score'])
    feature_imp_df_perm['Importance_Score'] = -1 * feature_imp_df_perm['Importance_Score']
    feature_imp_df_perm.loc[(feature_imp_df_perm.Importance_Score > 0), 'Importance'] = 'Positive'
    feature_imp_df_perm.loc[(feature_imp_df_perm.Importance_Score <= 0), 'Importance'] = 'Negative'
    feature_imp_df_perm = feature_imp_df_perm.sort_values(by=['Importance_Score'], ascending=False)
    return feature_imp_df_perm


def auto_feature_selection(model, X_train, y_train, X_test, y_test, rev_feature_list):
    model_ = clone(model)
    model_.random_state = 999
    model_.fit(X_train, y_train)
    baseline = mean_squared_error(y_test, model_.predict(X_test))
    for col in rev_feature_list:
        X = X_train.drop(col, axis=1)
        X_t = X_test.drop(col, axis=1)
        model_ = clone(model)
        model_.random_state = 999
        model_.fit(X, y_train)
        m = mean_squared_error(y_test, model_.predict(X_t))
        if m < baseline:
            feature = col
            break
    feature_index = rev_feature_list.index(feature)
    important_features = rev_feature_list[feature_index:]
    non_important_features = rev_feature_list[:feature_index]
    return important_features, non_important_features

