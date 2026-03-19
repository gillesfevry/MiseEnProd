import pandas as pd
import itertools
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

text_features = ["overview", "title"]
categorical_features = ["main_genre_name", "origin_country"]
numeric_features = ["budget", "popularity", "runtime", "vote_count", "vote_average", "timestamp"]
target = "revenue"

def create_random_forest_pipeline(n_estimators=100, max_depth = None, tfidf_max_features = 5000, tfidf_ngram_range = (1,1), tfidf_min_df=1, random_state = 0 ):

    def combine_text(X):
        return (X["overview"].fillna("") + " " + X["title"].fillna(""))

    text_pipeline = Pipeline([
        ("combine", FunctionTransformer(combine_text, validate=False)),
        ("tfidf", TfidfVectorizer(max_features = tfidf_max_features, stop_words="english", ngram_range = tfidf_ngram_range, min_df = tfidf_min_df))
    ])
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("text", text_pipeline, text_features),
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", RandomForestRegressor(n_estimators=n_estimators, max_depth = max_depth, random_state = random_state))
    ])

    return pipeline

def create_elastic_net_pipeline(alpha=1.0, l1_ratio=0.5, tfidf_max_features=1000,
                                 tfidf_ngram_range=(1,1), tfidf_min_df=1):
    def combine_text(X):
        return X["overview"].fillna("") + " " + X["title"].fillna("")

    text_pipeline = Pipeline([
       ("combine", FunctionTransformer(combine_text, validate=False)),
        ("tfidf", TfidfVectorizer(max_features=tfidf_max_features,
                                    stop_words="english",
                                   ngram_range=tfidf_ngram_range,
                                   min_df=tfidf_min_df))
    ])
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("text", text_pipeline, text_features),
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000))
    ])
    return pipeline

def model_cross_validation(data = None, pipeline = None, n_folds=10, scoring = "neg_root_mean_squared_error"):

    X = data[text_features + categorical_features + numeric_features]
    y = data[target]

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=n_folds,
        scoring=scoring,
        n_jobs=-1
    )

    return scores


def grid_search_random_forest(data, param_grid=None):
    """
    Teste différentes combinaisons d'hyperparamètres et retourne les résultats triés.
    """
    if param_grid is None:
        param_grid = {
            "n_estimators":       [100, 200],
            "max_depth":          [None],
            "tfidf_max_features": [3000],
            "tfidf_ngram_range":  [(1,1)],
            "tfidf_min_df":       [1],
        }

    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    print(f"{len(combinations)} combinaisons à tester...")
    results = []

    for combo in combinations:
        params = dict(zip(keys, combo))

        pipeline = create_random_forest_pipeline(
            n_estimators       = params["n_estimators"],
            max_depth          = params["max_depth"],
            tfidf_max_features = params["tfidf_max_features"],
            tfidf_ngram_range  = params["tfidf_ngram_range"],
            tfidf_min_df       = params["tfidf_min_df"],
        )

        scores = model_cross_validation(data=data, pipeline=pipeline)
        rmse   = (-scores).mean()
        std    = (-scores).std()

        print(f"RMSE: {rmse:,.0f} (+/- {std:,.0f}) | {params}")
        results.append({**params, "rmse_mean": rmse, "rmse_std": std})

    df_results = pd.DataFrame(results).sort_values("rmse_mean")
    return df_results

def grid_search_elastic_net(data, param_grid = None):

    if param_grid is None:
        param_grid = {
            "alpha":             [0.01, 0.1, 1.0, 10.0],
            "l1_ratio":          [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            "tfidf_max_features":[1000, 3000, 5000, 10000],
            "tfidf_ngram_range": [(1,1), (1,2)],
            "tfidf_min_df":      [1, 2, 5],
        }

    keys         = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))

    print(f"{len(combinations)} combinaisons à tester...")
    results = []

    for combo in combinations:
        params   = dict(zip(keys, combo))
        pipeline = create_elastic_net_pipeline(
            alpha              = params["alpha"],
            l1_ratio           = params["l1_ratio"],
            tfidf_max_features = params["tfidf_max_features"],
            tfidf_ngram_range  = params["tfidf_ngram_range"],
            tfidf_min_df       = params["tfidf_min_df"],
        )
        scores = model_cross_validation(data=data, pipeline=pipeline)
        rmse   = (-scores).mean()
        std    = (-scores).std()
        print(f"RMSE: {rmse:,.0f} (+/- {std:,.0f}) | {params}")
        results.append({**params, "rmse_mean": rmse, "rmse_std": std})

    df_results = pd.DataFrame(results).sort_values("rmse_mean")
    return df_results