from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def rfnum(X, y, n_estimators=100, test_size=0.2, random_state=None, report=True, importances=True):

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y = y_encoded

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if report:
        target_labels = label_encoder.classes_
        print("Rapport de classification :")
        print(classification_report(y_test, y_pred, target_names=target_labels, labels=np.arange(len(target_labels))))

    if importances: 
        importances = model.feature_importances_
        feature_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        print("Variables explicatives")
        print(feature_importance)

    return(model)

def rfnlp(X,y,n_estimators=100, test_size=0.2, random_state=None, report=True):

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y = y_encoded

    # Diviser les données en un ensemble d'entraînement et un ensemble de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Créer le TfidfVectorizer avec des paramètres plus stricts
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, min_df=2, max_df=0.95)

    # Appliquer le vectoriseur aux données d'entraînement
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Créer un modèle One-vs-Rest avec RandomForest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state)

    # Appliquer One-vs-Rest pour chaque genre
    ovr_model = OneVsRestClassifier(rf_model)
    ovr_model.fit(X_train_tfidf, y_train)

    if report:
        # Vérifier la précision du modèle sur les données de test
        y_pred = ovr_model.predict(X_test_tfidf)

        target_labels = label_encoder.classes_
        print("\nRapport de classification :")
        print(classification_report(y_test, y_pred, target_names=target_labels, labels=np.arange(len(target_labels))))
        
    return(ovr_model, vectorizer, label_encoder)

def plot_important_words_for_genre(importances, feature_names, genre_index, label_encoder, top_n=10):
    sorted_indices = np.argsort(importances)[::-1][:top_n]
    words = [feature_names[i] for i in sorted_indices]     
    scores = importances[sorted_indices]
        
    plt.figure(figsize=(10, 6))
    plt.barh(words, scores)
    plt.xlabel('Importance')
    plt.title(f'Mots les plus importants pour le genre {label_encoder.classes_[genre_index]}')
    plt.show()

def super_model(df, random_state=None):
    df1=df.copy()

    # Séparation des caractéristiques (X) et des labels (y)
    X = df1.drop(columns=['id','release_date', 'title', 'main_genre_name', "full_poster_path"])
    num=X.drop(columns=["overview"]).columns
    y = df1['main_genre_name']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y = y_encoded

    # Séparation en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Prétraitement pour les colonnes textuelles et numériques
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(), 'overview'),                 # Transformation TF-IDF pour les données textuelles
            ('num', StandardScaler(), num)  # Normalisation des données numériques
        ]
    )

    # Pipeline complet
    pipeline = Pipeline([
        ('preprocessor', preprocessor),  # Étape de prétraitement
        ('classifier', RandomForestClassifier(random_state=random_state))  # Modèle RFC
    ])

    # Entraîner le modèle
    pipeline.fit(X_train, y_train)

    # Prédictions et évaluation
    y_pred = pipeline.predict(X_test)

    target_labels = label_encoder.classes_

    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred, target_names=target_labels, labels=np.arange(len(target_labels))))

    return(pipeline)

def genreacp(df, n_components=2):

    dfacp=df.copy()

    #df1[df1['main_genre_id'].isin([ 18,35])]
    X=dfacp.drop(columns=['id','overview','release_date', 'title', 'main_genre_name', "full_poster_path", "revenue","budget",'main_genre_id'])
    y=dfacp['main_genre_id']

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)

    # Convertir le résultat en DataFrame
    X = pd.DataFrame(scaled_data, columns=X.columns)

    pca = PCA(n_components=n_components)  # n_components entre 2 et le nombre de features (ici 5) 
    x_2d = pca.fit_transform(X)


    # # Création d'un dataframe avec les données transformées:
    columns=['Composante {}'.format(i) for i in range(1, n_components + 1)]
    df_pca = pd.DataFrame(x_2d, columns=columns)
    df_pca['classe'] = y

    if n_components == 2:

        # # Visualiser les données PCA en 2D
        plt.figure(figsize=(8, 6))
        for classe in df_pca['classe'].unique():
            subset = df_pca[df_pca['classe'] == classe]
            plt.scatter(subset['Composante 1'], subset['Composante 2'], label=classe)

        plt.title("Visualisation des données après PCA (2 dimensions)")
        plt.xlabel("Composante 1")
        plt.ylabel("Composante 2")
        plt.legend()
        plt.show()

    return(df_pca)