from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def rfnum(X, y, n_estimators=100, test_size=0.2, random_state=None, report=True, importances=True):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if report:
        print("Rapport de classification :")
        print(classification_report(y_test, y_pred))

    if importances: 
        importances = model.feature_importances_
        feature_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)
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

def rftest(X,y,n_estimators=100, test_size=0.2, random_state=None, report=True):
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

    rf_model.fit(X_train_tfidf, y_train)

    if report:
        # Vérifier la précision du modèle sur les données de test
        y_pred = rf_model.predict(X_test_tfidf)

        target_labels = label_encoder.classes_
        print("\nRapport de classification :")
        print(classification_report(y_test, y_pred, target_names=target_labels, labels=np.arange(len(target_labels))))
        
    return(rf_model, vectorizer, label_encoder)

def plot_important_words_for_genre(importances, feature_names, genre_index, label_encoder, top_n=10):
    sorted_indices = np.argsort(importances)[::-1][:top_n]
    words = [feature_names[i] for i in sorted_indices]     
    scores = importances[sorted_indices]
        
    plt.figure(figsize=(10, 6))
    plt.barh(words, scores)
    plt.xlabel('Importance')
    plt.title(f'Mots les plus importants pour le genre {label_encoder.classes_[genre_index]}')
    plt.show()

def super_model(X,y):
    X = df[['text', 'numerical_feature']]
    y = df['label']

    # TF-IDF pour la colonne textuelle
    tfidf = TfidfVectorizer()

    # StandardScaler pour la colonne numérique
    scaler = StandardScaler()

    # Combiner les transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', tfidf, 'text'),  # Appliquer TF-IDF à la colonne 'text'
            ('num', scaler, ['numerical_feature'])  # Standardiser 'numerical_feature'
        ]
    )

    # Pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraîner le modèle
    model.fit(X_train, y_train)

    # Prédire sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Évaluer le modèle
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
