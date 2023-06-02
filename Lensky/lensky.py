import os
import tqdm
import itertools
import numpy as np
import nltk
from nltk import word_tokenize
import pymorphy2
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
from razdel import sentenize
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
from navec import Navec
from slovnet import Morph
from razdel import tokenize
import codecs
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
from wordcloud import WordCloud

LENSKY_STOP_WORDS = ('''я ты вы мы он она они себя кто что кто-то что-то кто-нибудь что-нибудь кто-либо 
что-либо кое-что кое-кто некто нечто никто ничто некого нечего ничего не ни тот та то те этот эта это эти такой
такая такое такие какие какое какой столько где как почему откуда зачем чего чему .. свой его её ее их ихний
э а и ну не нет да есть являться быть который которой которое твой мой наш ваш его ее её но на в перед после 
без через на над под из у около из под на между при с весь вся все всё так сам сама само мочь весь
вот всякий каждый каждая каждое какой-то к какой-то какая-то какое-то где-нибудь ” “ – казаться
сноска б для к после до или да но еще ещё это этот эта вот тот та те все всё тут там под над на
зато но хотя ли кто свой чем чей то''').split()


# Reads txt files in dict
def to_dict(data_path, files):
    out_dict = {}
    for file in files:
        with open(f'{data_path}\\{file}', encoding="utf-8") as f:
            out_dict[file[:-4]] = ' '.join(f.read().split())
    return out_dict


def dict_apply(func, corpus):
    keys = list(corpus.keys())
    out = []
    for author in keys:
        for book in corpus[author].keys():
            out.append(func(corpus[author][book]))
    return out


class Corpus:
    # {key1 : {key2 : obj}}
    # for example:
    # {writer1 : {book1 : 'text', book2 : ''text}...}

    def __init__(self, data_path=None, corpus=None):
        if corpus != None and type(corpus[list(corpus.keys())[0]]) is dict:
            self.corpus = corpus
        else:
            files = os.listdir(data_path)
            corpkeys = sorted(list(set([element.replace('_', '.').split('.')[0] for element in files])))
            self.corpus = {}
            print('Loading...')
            for key in tqdm.tqdm(corpkeys):
                key_books = []
                for i in range(len(files)):
                    if key == files[i].split('.')[0]:
                        key_books.append(files[i])
                self.corpus[key] = to_dict(data_path, key_books)
            print('Done!')

    def __getitem__(self, item):
        return self.corpus[item]

    def get(self, target):
        if len(target) == 1:
            if type(target[0]) == str:
                return self.corpus[target[0]]
            if type(target[0]) == int:
                return self.corpus[list(self.corpus.keys())[target[0]]]
        elif len(target) == 2:
            if type(target[0]) == str:
                return self.corpus[target[0]][target[1]]
            if type(target[0]) == int:
                key = list(self.corpus.keys())[target[0]]
                subkey = list(self.corpus[key].keys())[target[1]]
                return self.corpus[key][subkey]
        else:
            raise Exception("length of input target array should be < 3 and > 0")

    def keys(self):
        return list(self.corpus.keys())

    def values(self):
        return list(self.corpus.values())

    def subkeys(self):
        subkeys = []
        for key in self.corpus.keys():
            subkeys += self.corpus[key].keys()
        return subkeys

    def subvalues(self):
        subvalues = []
        for key in self.corpus.keys():
            subvalues += self.corpus[key].values()
        return subvalues

    def booknames(self):
        books = []
        for key in self.corpus.keys():
            for subkey in self.corpus[key].keys():
                books.append(' '.join((subkey.split('.')[1]).split('_')))
        return books

    def apply(self, func, replace=False):
        print('Processing...')
        keys = list(self.corpus.keys())
        if replace:
            for author in tqdm.tqdm(keys):
                for book in self.corpus[author].keys():
                    self.corpus[author][book] = func(self.corpus[author][book])
            print('Done!')
        else:
            out = {}
            for key in tqdm.tqdm(keys):
                out[key] = {}
                for book in self.corpus[key].keys():
                    out[key][book] = func(self.corpus[key][book])
            print('Done!')
            return Corpus(corpus=out)

    def save(self, data_path):
        print('Saving...')
        for key in tqdm.tqdm(self.corpus.keys()):
            for book in self.corpus[key]:
                with codecs.open(f'{data_path}/{book}.txt', 'w', 'utf-8') as file:
                    for word in self.corpus[key][book]:
                        file.write(f'{word}\n')
        print('Done!')

    def tokenize(self, stop_words=None, replace=False):
        stop_sym = set(['.', '...', '?', '!', ',', ';', '—', '«', '»', '[', ']',
                    '"', '-', ':', '`', '\'', '\\', '/', '(', ')', '``', "''",
                    '--', '*', '”' '“', '–', '..'])
        if stop_words != None:
            stop_sym += stop_words
        stop_sym = set(stop_sym)
        print('Tokenizing...')
        if replace:
            for key in tqdm.tqdm(self.keys()):
                for subkey in self[key].keys():
                    tokens = word_tokenize(self[key][subkey])
                    filtered = []
                    for word in tokens:
                        if word not in stop_sym and len(word) > 1:
                            filtered.append(word.lower())
                    self[key][subkey] = filtered
            print('Done!')
        else:
            out = {}
            for key in tqdm.tqdm(self.keys()):
                out[key] = {}
                for subkey in self[key].keys():
                    tokens = word_tokenize(self[key][subkey])
                    filtered = []
                    for word in tokens:
                        if word not in stop_sym and len(word) > 1:
                            filtered.append(word.lower())
                    out[key][subkey] = filtered
            print('Done!')
            return Corpus(corpus=out)

    def normalize(self, stop_words=None, replace=False):
        print('Normalizing...')
        if stop_words == None:
            stop_words = []
        stop_words = set(stop_words)
        morph = pymorphy2.MorphAnalyzer()
        if replace:
            for key in tqdm.tqdm(self.keys()):
                for subkey in self[key].keys():
                    tokens = self[key][subkey]
                    normalized = []
                    for word in tokens:
                        if word not in stop_words:
                            normalized.append(morph.parse(word)[0].normal_form)
                    self[key][subkey] = normalized
            print('Done!')
        else:
            out = {}
            for key in tqdm.tqdm(self.keys()):
                out[key] = {}
                for subkey in self[key].keys():
                    tokens = self[key][subkey]
                    normalized = []
                    for word in tokens:
                        normalized.append(morph.parse(word)[0].normal_form)
                    out[key][subkey] = normalized
            print('Done!')
            return Corpus(corpus=out)

    def low_pop_drop(self, lim=0.001, replace=False):
        print('Processing...')
        if replace:
            for author in tqdm.tqdm(list(self.keys())):
                for book in list(self[author].keys()):
                    text = self[author][book]
                    pop_series = pd.Series(text).value_counts(normalize=True).reset_index().to_numpy()
                    pop_set = set(pop_series[pop_series[:, 1] > lim][:, 0])
                    out = []
                    for word in text:
                        if word in pop_set:
                            out.append(word)
                    self[author][book] = out
            print('Done!')
        else:
            out = {}
            for author in tqdm.tqdm(list(self.keys())):
                out[author] = {}
                for book in list(self[author].keys()):
                    text = self[author][book]
                    pop_series = pd.Series(text.split()).value_counts(normalize=True).reset_index().to_numpy()
                    pop_set = set(pop_series[pop_series[:, 1] > lim][:, 0])
                    out[author][book] = []
                    for word in text.split():
                        if word in pop_set:
                            out[author][book].append(word)
                    out[author][book] = out
            print('Done!')
            return Corpus(corpus=out)

    def vectorize(self, vectorizer, replace=False):
        if type(self.get([0, 0])) == str:
            raise Exception("corpus subvalues should be lists like [word1, word2, ...]")
        # books = list(
        #     itertools.chain.from_iterable(self.subvalues()))
        books = []
        for element in self.subvalues():
            books.append(' '.join(element))
        # Vectorizing
        print('Vectorizer fitting...')
        vectorizer = vectorizer.fit(books)
        print('Done!')
        print('Vectorizing...')
        if replace:
            for key in tqdm.tqdm(self.keys()):
                for subkey in self[key].keys():
                    self[key][subkey] = Vector(vectorizer.transform([' '.join(self[key][subkey])]).A[0],
                                                 vectorizer.get_feature_names_out())  # sparse -> numpy
            print('Done!')
        else:
            out = {}
            for key in tqdm.tqdm(self.keys()):
                out[key] = {}
                for subkey in self[key].keys():
                    out[key][subkey] = Vector(vectorizer.transform([' '.join(self[key][subkey])]).A[0],
                                              vectorizer.get_feature_names_out())  # sparse -> numpy
            print('Done!')
            return Corpus(corpus=out)

    def nearest_neighbours(self, target, n, metric='minkowski', model=None):
        neigh = NearestNeighbors(n_neighbors=n + 1, metric=metric)
        vecs = []
        print('Processing...')
        for author in tqdm.tqdm(self.keys()):
            for book in list(self[author].keys()):
                vecs.append(self[author][book].vector)
        if model == None:
            neigh.fit(vecs)
            nearest_id = neigh.kneighbors([self.subvalues()[self.subkeys().index(target)].vector])[1]
            print('Done!')
            return np.array(self.subkeys())[nearest_id][0][1:]
        else:
            vecs = model.transform(vecs)
            neigh.fit(vecs)
            nearest_id = \
            neigh.kneighbors(model.transform([self.subvalues()[self.subkeys().index(target)].vector]))[1]
            print('Done!')
            return np.array(self.subkeys())[nearest_id][0][1:]



class Vector:
    def __init__(self, vector, keys):
        self.vector = vector
        self.keys = keys

    def __getitem__(self, item):
        return self.vector[item]

    def get_principal_words(self, num=None):
        if num == None:
            princ = np.argsort(self.vector)[::-1]
        else:
            princ = np.argsort(self.vector)[::-1][:num]
        return self.keys[princ]

    def get_principal_numbers(self, num=None):
        if num == None:
            princ = np.argsort(self.vector)[::-1]
        else:
            princ = np.argsort(self.vector)[::-1][:num]
        return self.vector[princ]


def vsum(vec_list):
    sum_ = np.zeros(len(vec_list[0].vector))
    for i in range(len(vec_list)):
        sum_ += vec_list[i].vector
    return Vector(sum_, vec_list[0].keys)


def vdif(a, b):
    return Vector(a.vector - b.vector, a.keys)


def vmean(vec_list):
    sum_ = np.zeros(len(vec_list[0].vector))
    for i in range(len(vec_list)):
        sum_ += vec_list[i].vector
    return Vector(sum_/len(vec_list), vec_list[0].keys)


def model_fit(corpus, model):
    list_ = []
    print('Loading data...')
    for author in tqdm.tqdm(corpus.keys()):
        for book in list(corpus[author].keys()):
            list_.append(corpus[author][book].vector)
    print('Done!')
    print('Model fitting...')
    model.fit(list_)
    print('Done!')


def plot_by_keys(corpus, model, colors=None, labels=None, point_size=7, alpha=0.5, title='TITLE',
                       legend_title='LEGEND_TITLE', grid=False, figsize=(10, 10), dpi=150):
    figure(figsize=figsize, dpi=dpi)
    if colors == None:
        colors = [f'C{i}' for i in range(len(corpus.keys()))]
    if labels == None:
        labels = [i for i in corpus.keys()]
    print('Corpus processing...')
    vecs = []
    for key in tqdm.tqdm(corpus.keys()):
        for subkey in list(corpus[key].keys()):
            vecs.append(corpus[key][subkey].vector)
    z_model = model.transform(vecs)
    start = 0
    stop = 0
    for i, key in tqdm.tqdm(enumerate(corpus.keys())):
        start = stop
        stop = stop + len(list(corpus[key].keys()))
        plt.scatter(z_model[start:stop, 0], z_model[start:stop, 1], s=point_size, color=colors[i],
                    alpha=alpha, label=labels[i])
    plt.title(title)
    plt.legend(title=legend_title, loc='upper center', bbox_to_anchor=(0.5, -0.05))
    plt.xticks([])
    plt.yticks([])
    if grid:
        plt.grid()
    print('Done!')


def plot_by_subkeys(corpus, target, model, colors=None, labels=None, point_size=7, alpha=0.5, title='TITLE',
                    legend_title=None, grid=False, figsize=(10, 10), dpi=150):
    figure(figsize=figsize, dpi=dpi)
    if colors == None:
        colors = [f'C{0}' for i in range(len(corpus.keys()))]
    if labels == None:
        labels = [i for i in list(corpus[target].keys())]
    if legend_title == None:
        legend_title = target
    print('Corpus processing...')
    vecs = []
    for author in tqdm.tqdm(corpus.keys()):
        for book in list(corpus[author].keys()):
            vecs.append(corpus[author][book].vector)
    z_model = model.transform(vecs)
    plt.scatter(z_model[:, 0], z_model[:, 1], color='gray', s=7, alpha=0.3)
    print('Done!')
    print('Target values processing...')
    author_vecs = []
    for book in list(corpus[target].keys()):
        author_vecs.append(corpus[target][book].vector)
    z_model = model.transform(author_vecs)
    for i, book in tqdm.tqdm(enumerate(corpus[target])):
        plt.scatter(z_model[i, 0], z_model[i, 1], label=' '.join((labels[i].split('.')[1]).split('_')), s=point_size,
                    color=colors[i], marker=f'${i+1}$')
    print('Done!')
    plt.title(title)
    plt.legend(title=legend_title, loc='upper center', bbox_to_anchor=(0.5, -0.05))
    plt.xticks([])
    plt.yticks([])
    if grid:
        plt.grid()


def plot_interactive(corpus, model, data_key='key', data_subkey='subkey'):
    print('Corpus processing...')
    vecs = []
    df = {data_key : [], data_subkey : [], 'z0' : [], 'z1' : []}
    for key in tqdm.tqdm(corpus.keys()):
        for subkey in list(corpus[key].keys()):
            vecs.append(corpus[key][subkey].vector)
            df[data_key].append(key)
            df[data_subkey].append(subkey)
    z_model = model.transform(vecs)
    df['z0'] = z_model[:, 0]
    df['z1'] = z_model[:, 1]
    df = pd.DataFrame(data=df)
    fig = px.scatter(df, x='z0', y='z1', color=data_key, hover_data=[data_subkey])
    fig.show()


def moving_window(a, n):
    ret = []
    for i in range(len(a)):
        if i < n:
            win = a[:i+n+1]
        elif len(a)-i-1 < n:
            win = a[i-n:]
        else:
            win = a[i-n:i+n+1]
        ret.append(np.mean(win))
    return ret


def sentiment_plot(book, window=1, plot_type=None, title='TITLE', zero_level=True, grid=True, figsize=(12, 6), dpi=150):
    figure(figsize=figsize, dpi=dpi)
    if plot_type == None:
        sentiments = ['neutral', 'positive', 'negative', 'speech']
    if type(plot_type) == list:
        sentiments = plot_type
    if plot_type in ['sentiment', 'pos-neg', 'comb']:
        sentiments = ['positive', 'negative']
    tokenizer = RegexTokenizer()
    model = FastTextSocialNetworkModel(tokenizer=tokenizer)
    substrings = list(sentenize(book))
    sents = []
    for i in range(len(substrings)):
        sents.append(substrings[i].text)
    messages = sents
    results = model.predict(messages, k=2)
    sentiment_dict = {'neutral' : [], 'positive' : [],
                      'negative' : [], 'speech' : []}
    for message, sentiment in zip(messages, results):
        for s in sentiments:
            if s not in sentiment.keys():
                sentiment[s] = 0
            sentiment_dict[s].append(sentiment[s])
    if type(plot_type) == list or plot_type == None:
        for s in sentiments:
            result = moving_window(sentiment_dict[s], window)
            plt.plot(result, label=s)
            plt.xlabel('Процент от текста')
            plt.xticks([i*len(result)/10 for i in range(11)], [i*10 for i in range(11)])
            plt.ylabel('Интенсивность')
            plt.legend(title='Метка', loc="upper left", bbox_to_anchor=(1,1))
            if zero_level:
                plt.plot([0, len(result)], [0, 0], '--', color='grey')
        if grid:
                plt.grid()
        plt.title(title)
    elif type(plot_type) == str:
        if plot_type == 'sentiment':
            result = moving_window(0.5*(np.array(sentiment_dict['positive']) + np.array(sentiment_dict['negative'])), window)
            plt.plot(result)
            plt.xlabel('Процент от текста')
            plt.xticks([i*len(result)/10 for i in range(11)], [i*10 for i in range(11)])
            plt.ylabel('Сентиментальность')
            plt.title(title)
            if zero_level:
                plt.plot([0, len(result)], [0, 0], '--', color='grey')
            if grid:
                plt.grid()
        elif plot_type == 'pos-neg':
            result = moving_window(np.array(sentiment_dict['positive']) - np.array(sentiment_dict['negative']), window)
            plt.plot(result)
            plt.xlabel('Процент от текста')
            plt.xticks([i*len(result)/10 for i in range(11)], [i*10 for i in range(11)])
            plt.ylabel('Тональность')
            plt.title(title)
            if zero_level:
                plt.plot([0, len(result)], [0, 0], '--', color='grey')
            if grid:
                plt.grid()
        elif plot_type == 'comb':
            result = moving_window(np.array(sentiment_dict['positive']) - np.array(sentiment_dict['negative']), window)
            plt.plot(result, label='Тональность')
            result = moving_window(np.array(sentiment_dict['positive']) + np.array(sentiment_dict['negative']), window)
            plt.plot(result, label='Сентименатальность')
            plt.legend()
            plt.xlabel('Процент от текста')
            plt.xticks([i*len(result)/10 for i in range(11)], [i*10 for i in range(11)])
            plt.ylabel('Тональность/Сентиментальность')
            plt.title(title)
            if zero_level:
                plt.plot([0, len(result)], [0, 0], '--', color='grey')
            if grid:
                plt.grid()
        else:
            raise Exception("str type of plot_type should be in ['sentiment', 'pos-neg', 'comb']")
    else:
        raise Exception("plot_type should be str or list")


def morph_plot(book, window, plot_type, title='TITLE', grid=True, figsize=(12, 6), dpi=150):
    figure(figsize=figsize, dpi=dpi)
    if type(plot_type) != list:
        raise Exception("plot_type should be list")
    navec = Navec.load('models//navec_news_v1_1B_250K_300d_100q.tar')
    morph = Morph.load('models//slovnet_morph_news_v1.tar')
    morph.navec(navec)
    m = {'ADJ' : [], 'ADP' : [], 'ADV' : [], 'AUX' : [], 'CCONJ' : [],
        'DET' : [], 'INTJ' :[], 'NOUN' : [], 'NUM' : [], 'PART' : [], 'PRON' : [],
        'PROPN' : [], 'PUNCT' : [], 'SCONJ' : [], 'SYM' : [], 'VERB' : [], 'X' : []}
    tokens = morph([x.text for x in list(tokenize(book))]).tokens
    for token in tokens:
        res = token.pos
        for key in m.keys():
            if key == res:
                m[key].append(1)
            else:
                m[key].append(0)
    for morph in plot_type:
        result = moving_window(m[morph], window)
        plt.plot(result, label=morph)
        plt.xlabel('Процент от текста')
        plt.ylabel('Доля части речи')
        plt.xticks([i*len(result)/10 for i in range(11)], [i*10 for i in range(11)])
        plt.legend(title='Часть речи', loc="upper left", bbox_to_anchor=(1,1))
    plt.title(title)
    if grid:
        plt.grid()


def plot_wordcloud(book, stopwords=[]):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    wordcloud = WordCloud(width= 3000, height = 2000, random_state=1, background_color='salmon', colormap='Pastel1', collocations=False, stopwords=stopwords).generate(book)
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off")

