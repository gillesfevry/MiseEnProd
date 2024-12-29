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
    """
        Uses numerical data to predict to which category an item pretends. Not specific to TMDB datframe. 

        Parameters:
        X: a pandas data-frame with only numerical data
        y: a pandas series of the same length with the name of the categories we are trying to predict.
        n_estimators: an int, the number of estimators used in the RF model
        test_size: a number between 0 and 1, the portion of the df on which the model will be tested
        random_state: the random seed
        report: a boolean, print a report or not
        importances: a boolean, print a recap of the importance of the variables or not

        Returns:
        df1: a random forest model.
    """

    #encode y
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    #split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    #train model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    if report:
        y_pred = model.predict(X_test)
        target_labels = label_encoder.classes_
        print("Rapport de classification :")
        print(classification_report(y_test, y_pred, target_names=target_labels, labels=np.arange(len(target_labels))))

    if importances: 
        importances = model.feature_importances_
        feature_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        print("Variables explicatives")
        print(feature_importance)

    return(model)

def rfnlp(X,y,n_estimators=100, test_size=0.2, random_state=None, report=True, stop_words='english', max_features=1000, min_df=2, max_df=0.95 ):
    """
        Uses a text in english to predict to which category an item pretends, using a TF-IDF vectorizer. Not specific to TMDB datframe. 

        Parameters:
        X: a pandas series of texts
        y: a pandas series of the same length with the name of the categories we are trying to predict.
        n_estimators: an int, the number of estimators used in the RF model
        test_size: a number between 0 and 1, the portion of the df on which the model will be tested
        random_state: the random seed
        report: a boolean, print a report or not
        stop_words: a str, the language of the text to use to stop uncessary words for the tfidf vectorizer
        max_features: the maximum number of words we want to use
        min_df: the minimum number of times a word must be present
        max_df: the maximal portion of textes with a word

        Returns:
        ovr_model: a one versus all random forest model.
        vectorizer: the vectorizer trained on our data
        label_encoder: a label encoder fited on our categories
    """

    #encode y
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # create the tf-idf vectorizer and apply it to our data
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=max_features, min_df=min_df, max_df=max_df)

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Create a One-vs-Rest model with RandomForest, this will be usefull to see which words help predict each category
    rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state)

    # apply the One-vs-Rest for each category genre
    ovr_model = OneVsRestClassifier(rf_model)
    ovr_model.fit(X_train_tfidf, y_train)

    if report:
        y_pred = ovr_model.predict(X_test_tfidf)
        target_labels = label_encoder.classes_
        print("\nRapport de classification :")
        print(classification_report(y_test, y_pred, target_names=target_labels, labels=np.arange(len(target_labels))))
        
    return(ovr_model, vectorizer, label_encoder)

def plot_important_words_for_genre(list=None, ovr_model=None, vectorizer=None, label_encoder=None, top_n=10):
    """
        Uses the results of the rfnlp function to plot the most important words for each category

        Parameters:
        list: a list of ints, between 0 and len(ovr_model.estimators_), corresponding to which genres will be ploted
        ovr_model: a one versus all random forest model, eg the one returned by rfnlp
        vectorizer: the vectorizer trained on our data, eg the one returned by rfnlp
        label_encoder: a label encoder fited on our categories, eg the one returned by rfnlp
        top_n: number of words shown per plot

        Plots:
        A graph for each category with the most important words to predict this category
    """
    for i in list:

        sorted_indices = np.argsort(ovr_model.estimators_[i].feature_importances_)[::-1][:top_n]
        words = [vectorizer.get_feature_names_out()[i] for i in sorted_indices]     
        scores = ovr_model.estimators_[i].feature_importances_[sorted_indices]
            
        plt.figure(figsize=(10, 6))
        plt.barh(words, scores)
        plt.xlabel('Importance')
        plt.title(f'Mots les plus importants pour le genre {label_encoder.classes_[i]}')
        plt.show()

def super_model(df, random_state=None): #with the help of ChatGPT
    #this function is coded specifically to our data and can not be used in other cases. 
    #it is coded as if it was part of a Notebook but written here just for clarity

    """
        Uses numerical data and a text in english to predict to which category an item pretends, 
        using a random forest pipeline and a TF-IDF vectorizer.

        Parameters:
        df: a pandas data-farme provided by get_infos and cleaned
        random_state: the random seed

        Returns:
        pipeline: a random forest pipeline. 
    """

    df1=df.copy()

    # Separating numerical and textual information
    X = df1.drop(columns=['id','release_date', 'title', 'main_genre_name', "full_poster_path"])
    num=X.drop(columns=["overview"]).columns
    y = df1['main_genre_name']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Spliting train and test samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Preprocessing fot numerical and textual columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(), 'overview'), # TTF-IDF transformation for textual data
            ('num', StandardScaler(), num)  # Normalising of numerical data
        ]
    )

    # Complete Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),  # preteatment
        ('classifier', RandomForestClassifier(random_state=random_state))  # RFC model
    ])

    # training
    pipeline.fit(X_train, y_train)

    # prediction and evaluation
    y_pred = pipeline.predict(X_test)

    target_labels = label_encoder.classes_

    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred, target_names=target_labels, labels=np.arange(len(target_labels))))

    return(pipeline)

def genreacp(df, n_components=2):
    #this function is coded specifically to our data and can not be used in other cases. 
    #it is coded as if it was part of a Notebook but written here just for clarity
    """
        Creates an ACP of the movie genres

        Parameters:
        df: a pandas data-farme provided by get_infos and cleaned
        n_components: the number of main components for the ACP, graph will only be plotted for =2

        Returns:
        df_pca: a pandas data-frame with the coordonates of each movie in the PCA
    """
    dfacp=df.copy()

    #keep only useful data
    X=dfacp.drop(columns=['id','overview','release_date', 'title', 'main_genre_name', "full_poster_path", "revenue","budget",'main_genre_id'])
    y=dfacp['main_genre_id']

    #scale data and convert it to a df
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)

    X = pd.DataFrame(scaled_data, columns=X.columns)

    #PCA
    pca = PCA(n_components=n_components)  
    x = pca.fit_transform(X)

    # Dataframe with transformed data:
    columns=['Composante {}'.format(i) for i in range(1, n_components + 1)]
    df_pca = pd.DataFrame(x, columns=columns)
    df_pca['classe'] = y

    #visulaize PCA in 2D
    if n_components == 2:

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