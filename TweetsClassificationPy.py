import json
import glob
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
from sklearn.decomposition import TruncatedSVD


def load_data():
    list = []
    for f in glob.glob("*.json"):   #loading only json files
        with open(f, "rb") as infile:
            list += (json.load(infile))     #combing all the data to a big list
    return pd.DataFrame(list)       #convert the list to a pandas datarame

    #Clearning the text by tokening the words, removing all non letter characters, stop words and using lemmaization
def clean_text(text):
    tokenzier = nltk.tokenize.casual.TweetTokenizer(strip_handles=True, reduce_len=True)
    text = tokenzier.tokenize(text)
    text = [word for word in text if word.isalpha()]
    text = [word for word in text if not word in stopwords.words("english")]
    lemma = nltk.stem.WordNetLemmatizer()
    text = [lemma.lemmatize(w, pos="v") for w in text]  #for verbs
    return [lemma.lemmatize(w, pos="n") for w in text]  #for nouns

def preprocess_old_tweets(df):
    print("Dataset and tweets*before* preprocessing and text normalization:\n", df.text.head(5))
    df.info()
    df['date'] = pd.to_datetime(df['created_at'], errors='coerce')  # changing data to pd.datatime
    df['day'] = df['date'].dt.day_of_week  # taking the day of tweet as a feature
    df['hour'] = df['date'].dt.hour  # taking the hour of tweet as a feature
    entities = pd.json_normalize(df['entities'])  # Changing dictionary to columns
    user = pd.json_normalize(df['user'])
    user_ent = user['entities.url.urls'].apply(pd.Series)
    user_entities_url = pd.json_normalize(user_ent[0].fillna(method='pad'))
    pdList = [df, user, entities, user_entities_url]
    df = pd.concat(pdList, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df.drop(['date', 'created_at', 'user', 'entities', 'entities.url.urls', 'indices'], axis=1, inplace=True)
    df = df.loc[:, df.isnull().mean() < 0.4]  # Removing features with too much missing values
    for col in df.columns:
        if type(df[col][0]) == list and df[col].astype(
                bool).mean() < 0.7:  # check if there is a list and if that list has only repeated values
            df.drop([col], axis=1, inplace=True)
    df = df.loc[:,
         df.nunique() != len(df)]  # Removing features that have the have different value for each data points
    df = df.loc[:, df.nunique() != 1]  # Removing features that have only one value
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)  # lower case all the text
    df['source'] = np.where(df['source'].str.contains("android"), 2,
                            (np.where(df['source'].str.contains("iphone"), 1, 0)))  # chaning values to numbers
    tqdm.pandas(desc='clean_old_tweets_progress')  # giving a progress bar for the next (slow) function
    df['text'] = df['text'].progress_apply(clean_text)  # clean text
    print("\nDataset and tweets *after* preprocessing and text normalization:\n", df.text.head(5))
    df.info()
    return df

    # choosing only android (1) and iphone(0) devices:
def filter_cell_phones(x, y):
    df = pd.concat([x, y], axis=1)
    x_cells = df[df['source'] > 0]      # only cell phones
    x_other = df[df['source'] == 0]     # other_devices
    y_cells = x_cells.source
    y_other = x_other.source
    x_cells = x_cells.drop(['source'], axis=1)
    x_other = x_other.drop(['source'], axis=1)
    return x_cells, y_cells, x_other, y_other


    # checking if the data is unbalance and fixing it
def imbalance_data(X, y):
    im1 = sns.countplot(y)
    im1.set_xticklabels(['iPhone', 'Android'])
    plt.show()
    under_sample = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = under_sample.fit_resample(X, y)
    return X_resampled, y_resampled


    #Creating a dataframe by fitting the data and the tfidf (on the text column) so it will run on the model
def tfidf(df):
    y = df.source
    X = df.drop(['source'], axis=1)
    X_oh = pd.get_dummies(X[X.columns.difference(['text'])], drop_first=True) # one hot vectors on all the data except the text and source (labels)
    X_oh.insert(0, "text", X.text)
    X_oh.text = [" ".join(tokens) for tokens in X_oh.text.values] #changing the lists to strings
    x_cells, y_cells, x_other, y_other = filter_cell_phones(X_oh, y)
    X_balanced, y_balanced = imbalance_data(x_cells, y_cells)
    tfv = TfidfVectorizer(min_df=3, max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                          stop_words='english')
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.25, random_state=0)
    tfv.fit(X_train['text'])
    X_train, train_sm = sparse_matrix_fit(X_train, tfv)
    X_test, test_sm = sparse_matrix_fit(X_test, tfv)
    x_other, _ = sparse_matrix_fit(x_other, tfv)

    #svd = TruncatedSVD(n_components=500, n_iter=7, random_state=0)     #reducing the number of features to 500 so it wont have more features than data points
    #X = svd.fit_transform(X)   # it reduced the AUC by 2% so there was no point.
    return X_train, X_test, y_train, y_test, x_other, y_other, tfv, train_sm, test_sm


def sparse_matrix_fit(X, tfv):
    sparse_mat = pd.DataFrame.sparse.from_spmatrix(tfv.transform(X['text']))  # make the sparse matrix fit the pandas df
    X = pd.concat([X.reset_index(), sparse_mat.reset_index()], axis=1).drop(['text', 'index'], axis=1)
    X.columns = [str(i) for i in range(0, (X.shape[1]))]  # changing columns names so it two columns wont share a name
    return X, sparse_mat


def xgb_model(X_train, X_test, y_train, y_test):
    xgboost = xgb.XGBClassifier(nthread=-1, eval_metric='auc', random_state=0, enable_categorical=True)
    xgboost.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    y_pred = xgboost.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("AUC score:", score)
    return xgboost, score


def model_2016(x_other, xgboost):
    y_pred = pred_model(x_other, xgboost)
    print(f'The classifier found that Trump wrote {y_pred.value_counts().values[0]} tweets from the other devices'
          f' while the unknown assistant wrote {y_pred.value_counts().values[1]} tweets from the other devices')


def pred_model(x, xgboost):
    return pd.Series(xgboost.predict(x))


def preprocess_2018_tweets(df):
    print("Dataset and tweets*before* preprocessing and text normalization:\n", df.text.head(5))
    df.info()
    df.text = df.text.fillna(df.full_text)
    df['date'] = pd.to_datetime(df['created_at'], errors='coerce') #changing data to pd.datatime
    df['day'] = df['date'].dt.day_of_week   #taking the day of tweet as a feature
    df['hour'] = df['date'].dt.hour     #taking the hour of tweet as a feature
    entities = pd.json_normalize(df['entities'])
    user = pd.json_normalize(df['user'])
    pdList = [df, user, entities]
    df = pd.concat(pdList, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df.drop(['date', 'created_at', 'user', 'entities', 'full_text'], axis=1, inplace=True)
    df = df.loc[:, df.isnull().mean() < 0.4]    # Removing features with too much missing values
    for col in df.columns:
        if type(df[col][0]) == list and df[col].astype(bool).mean() < 0.7:
            df.drop([col], axis=1, inplace=True)
    df = df.loc[:, df.nunique() != len(df)]     #Removing features that have the have different value for each data points
    df = df.loc[:, df.nunique() != 1]       #Removing features that have only one value
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)      #lower case all the text
    df['source'] = np.where(df['source'].str.contains("android"), 2, (np.where(df['source'].str.contains("iphone"), 1, 0))) #chaning values to numbers
    tqdm.pandas(desc='clean_text_progress')     #giving a progress bar for the next (slow) function
    df['text'] = df['text'].progress_apply(clean_text) # clean text
    print("\nDataset and tweets *after* preprocessing and text normalization:\n", df.text.head(5))
    df.info()
    return df


def model_2018(df, xgboost, tfv):
    text = [" ".join(tokens) for tokens in df['text'].values] #changing the lists to strings
    sparse_mat = pd.DataFrame.sparse.from_spmatrix(tfv.transform(text))
    sparse_mat.columns = [str(i) for i in range(0, (sparse_mat.shape[1]))]
    y_pred = pred_model(sparse_mat, xgboost_text_model)
    print(f'The classifier found that Trump wrote {y_pred.value_counts().values[0]} tweets in 2018 '
          f' while the unknown assistant wrote {y_pred.value_counts().values[1]} tweets in 2018')


if __name__ == '__main__':

    df = load_data()
    df = preprocess_old_tweets(df)
    X_train, X_test, y_train, y_test, x_other, y_other, tfv, train_sm, test_sm = tfidf(df)      #tfidf fitting
    xgboost, metadata_score = xgb_model(X_train, X_test, y_train, y_test)   #model based on all the data
    xgboost_text_model, text_score = xgb_model(train_sm, test_sm, y_train, y_test)      #model based only on the text using tfidf
    model_2016(x_other, xgboost)

    df2 = pd.read_json(r'master_2018')      #2018 tweets
    df2 = preprocess_2018_tweets(df2)       #preprocess_2018_tweets we found that they features have changed since 2016
                                            #so the metadata model cant be used, only model based on text.
    model_2018(df2, xgboost_text_model, tfv)






    # xgb_all_devices(xgboost, X_test, y_test, x_other, y_other)

# def xgb_all_devices(xgboost, X_test, y_test, x_other, y_other):
#     x_all_devices = pd.concat([X_test, x_other])
#     y_all_devices = pd.concat([y_test, y_other])
#     idx = np.random.permutation(len(x_all_devices))
#     x, y = x_all_devices[idx], y_all_devices[idx]
#     im2 = sns.countplot(y_all_devices)
#     im2.set_xticklabels(['iPhone', 'Android', 'Others'])
#     plt.show()
#     under_sample = RandomUnderSampler(random_state=0)
#     X_resampled, y_resampled = under_sample.fit_resample(x, y)
#     y_pred = xgboost.predict(X_resampled)
#     print("AUC score for all devices:", accuracy_score(y_resampled, y_pred))



