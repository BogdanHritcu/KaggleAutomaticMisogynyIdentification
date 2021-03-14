import nltk
import pandas as pd
import numpy as np
from collections import Counter
import random
import re
from nltk.corpus import stopwords
import time
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


def tokenize(text):
    """
    :param text: Textul care trebuie impartit in tokens
    :return: Lista filtrata de tokens
    """
    tokens = nltk.TweetTokenizer(preserve_case=False, reduce_len=True).tokenize(text)
    tokens = [re.sub(r'http\S+', '_URL_', token) for token in tokens]  # inlocuieste 'http://www.site.com' cu '_URL_'
    tokens = [re.sub(r'\A@{1}\S+', '_TAG_A_', token) for token in tokens]  # inlocuieste '@utilizator' cu '_TAG_A_'
    tokens = [re.sub(r'\A#{1}\S+', '_TAG_D_', token) for token in tokens]  # inlocuieste '#cuvant' cu '_TAG_D_'
    tokens = [token for token in tokens if len(re.findall(r'[^\w\s]', token)) == 0]  # inlatura semnele de punctuatie
    tokens = [word for word in tokens if word not in stopwords.words('italian')]  # inlatura stopwords (prepozitii)

    return tokens


def get_vocabulary(corpus, labels = None):
    """
    :param corpus: o lista care contine toate textele
                   din care vrem sa construim vocabularul
    :return: un counter care contine fiecare cuvant din vocabular
             impreuna cu numarul de aparitie al acestuia in corpus
             (
                functia folosita pentru tokenizare poate returna si ngrame,
                deci vocabularul ar putea contine si ngrame, in functie de tokenizare
             )
    """
    if labels is None:
        counter = Counter()
        for text in corpus:
            tokens = tokenize(text)
            counter.update(tokens)
        return counter
    else:
        classes = np.unique(labels)
        corpus_classes = np.array([corpus[labels == cls] for cls in classes])
        counters = [Counter() for cls in classes]
        for idx, corpus_cls in enumerate(corpus_classes):
            for text in corpus_cls:
                tokens = tokenize(text)
                counters[idx].update(tokens)
        return counters


def get_representation(vocabulary, how_many):
    """
    :param vocabulary: Un counter care contine toate cuvintele din vocabular
                       (posibil si ngrame) si numarul lor de aparitie in corpus
    :param how_many: Dimensiunea vectorului de caracteristici (se foloseste
                     pentru a obtine primele how_many cele mai comune cuvinte
                     sau ngrame din vocabular)
    :return: Un dictionar care contine cuvintele din noua reprezentare alaturi
             de indexul la care se afla acestea in reprezentare (word2idx)
             si un dictionar care contine indexurile din reprezentare alaturi
             de cuvintele de la acele indexuri
    """

    most_comm = vocabulary.most_common(how_many)
    word2idx = {}
    idx2word = {}
    for idx, itr in enumerate(most_comm):
        word = itr[0]
        word2idx[word] = idx
        idx2word[idx] = word

    return word2idx, idx2word


def text_to_bow(text, word2idx):
    """
    :param text: Textul care trebuie transormat intr-un vector de caracteristici
    :param word2idx: Un dictionar care contine cuvintele din vocabular si indexurile
                     la care trebuie asezate in reprezentare (vectorul de caracteristici)
    :return: Vectorul de caracteristici obtinut din text
    """
    features = np.zeros(len(word2idx))
    for token in tokenize(text):
        if token in word2idx:
            features[word2idx[token]] += 1
    return features


def corpus_to_bow(corpus, word2idx):
    """
    :param corpus: Lista cu toate textele
    :param word2idx: Un dictionar care contine cuvintele din vocabular si indexurile
                     la care trebuie asezate in reprezentare (vectorul de caracteristici)
    :return: O lista cu vectorii de caracteristici corespunzatori textelor din corpus
    """
    all_features = np.array([text_to_bow(text, word2idx) for text in corpus])

    return all_features


def write_prediction(out_file, predictions):
    """
    :param out_file: Numele fisierului de in care sunt scrise predictiile
    :param predictions: Lista de predictii care trebuie scrise in fisier
    """
    with open(out_file, 'w') as fout:
        fout.write('id,label\n')  # scrie capul de tabel
        start_id = 5001  # exemplele de testare incep de la id-ul 5001
        for i, pred in enumerate(predictions):
            linie = str(start_id + i) + ',' + str(pred) + '\n'
            fout.write(linie)


def split_data(data, labels, percent_test=0.25):
    """
    :param data: Lista cu toti vectorii de caracteristici
    :param labels: Lista cu toate etichetele corespunzatoare vectorilor de caracteristici
    :param percent_test: Procentul de din date si etichete care va fi folosit ca date
                         si etichete de test
    :return: Lista cu vectorii de caracteristici pentru datele de antrenare
             Lista cu vectorii de caracteristici pentru datele de test
             Lista cu etichetele pentru datele de antrenare
             Lista cu etichetele pentru datele de test
    """
    indices = np.arange(0, len(labels))
    random.shuffle(indices)
    N = int((1.0 - percent_test) * len(labels))
    train = data[indices[:N]]
    test = data[indices[N:]]
    y_train = labels[indices[:N]]
    y_test = labels[indices[N:]]

    return train, test, y_train, y_test


def k_folds_split(k, data, labels):
    """
    :param k: Numarul de parti in care sectionam multimea de date (folds)
    :param data: Lista cu toti vectorii de caracteristici
    :param labels: Lista cu toate etichetele corespunzatoare vectorilor de caracteristici
    :yield: Lista cu vectorii de caracteristici pentru datele de antrenare (combinata din k-1 folds)
            Lista cu vectorii de caracteristici pentru datele de test (1 fold)
            Lista cu etichetele pentru datele de antrenare (combinata din k-1 folds)
            Lista cu etichetele pentru datele de test (1 fold)
    """
    chunk_size = len(labels) // k
    indices = np.arange(0, len(labels))
    random.shuffle(indices)

    for i in range(0, len(labels), chunk_size):
        train_indices = np.concatenate([indices[0:i], indices[i + chunk_size:]])
        test_indices = indices[i:i + chunk_size]
        train = data[train_indices]
        test = data[test_indices]
        y_train = labels[train_indices]
        y_test = labels[test_indices]

        yield train, test, y_train, y_test


class MultinomialNaiveBayes:
    def __init__(self, smooth=1.0, prior=None):
        if prior is None:
            prior = [0.5, 0.5]
        self.smooth = smooth
        self.freq = None
        self.prior = prior
        self.lkhoods = None

    def fit(self, train_data, train_labels):
        total_train_no = train_data.shape[0]

        train_classes = np.array([train_data[train_labels == cls] for cls in np.unique(train_labels)])

        self.freq = np.array([train_cls.sum(axis=0) for train_cls in train_classes]) + self.smooth

        self.lkhoods = self.freq / self.freq.sum(axis=1).reshape(-1, 1)
        self.prior = np.array([len(train_cls) / total_train_no for train_cls in train_classes]) if self.prior is None \
            else np.array(self.prior)

    def get_scores(self, test_data):
        scores = np.zeros(shape=(test_data.shape[0], self.prior.shape[0]))
        for i, test in enumerate(test_data):
            present_features = test.astype(bool)
            present_lkhoods = self.lkhoods[:, present_features] * test[present_features]
            lkhood_scores = np.log(present_lkhoods).sum(axis=1)
            scores[i] = lkhood_scores + np.log(self.prior)
        return scores

    def predict(self, test_data):
        pred = self.get_scores(test_data).argmax(axis=1)
        return pred


# Obtine datele de antrenament si de test alaturi de forma (index, text)
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Construieste corpusul de antrenament si cel de test
train_corpus = train_df['text'].values
test_corpus = test_df['text'].values

# Obtine etichetele datelor de antrenament
train_labels = train_df['label'].values

# Obtine vocabularul format din toate cuvintele din corpusul de antrenament
vocabulary = get_vocabulary(train_corpus)

# Obtine noul vocabular fomat din cele mai comune N cuvinte din vocabularul complet
word2idx, idx2word = get_representation(vocabulary, 4000)

# Obtine datele de test si de antrenament sub forma de vectori de caracteristici
# (forma este data de reprezentarea vocabularului de mai sus)
train_data = corpus_to_bow(train_corpus, word2idx)
test_data = corpus_to_bow(test_corpus, word2idx)

# 10-folds cross validation
scores = []
conf_mats = []
clf = MultinomialNaiveBayes(smooth=0.5)
for trn_data, tst_data, trn_labels, tst_labels in k_folds_split(10, train_data, train_labels):
    clf.fit(trn_data, trn_labels)
    pred = clf.predict(tst_data)
    score = f1_score(y_true=tst_labels, y_pred=pred)
    scores.append(score)
    conf_mat = confusion_matrix(y_true=tst_labels, y_pred=pred)
    conf_mats.append(conf_mat)

print('Scorul mediu este(F1):', np.mean(scores))
print('Matricea de confuzie este:', np.sum(conf_mats, axis=0))

# Creeaza un clasificator Multinomial Naive Bayes si seteaza variabila smooth la 0.5
clf = MultinomialNaiveBayes(smooth=0.5)
# Antreneaza perceptronul
clf.fit(train_data, train_labels)
# Obtine predictiile
pred = clf.predict(test_data)
# Scrie predictiile intr-un fisier
write_prediction("multinomial_nb.csv", pred)
