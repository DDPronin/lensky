# Lensky: Russian fiction literature analysis toolkit


![alt tag](https://github.com/DDPronin/lensky/blob/main/Lensky_badge.png)

Lensky - проект с открытым исходным кодом, включающий в себя корпус из 750+ произведений русской литературы XVIII-XXI веков, набор функций для их анализа и примеры. Для корректной работы функций требуется, чтобы на компьютере пользователя были установлены следующие python библиотеки: tqdm, numpy, nltk, pymorphy2, matplotlib, dostoevsky, natasha, pandas, sklearn, plotly, wordcloud.

<br/> Для установки необходимых библиотек воспользуйтесь следующими командами:

```bash
$ pip install tqdm
```
```bash
$ pip install numpy
```
```bash
$ pip install nltk
```
```bash
$ pip install matplotlib
```
```bash
$ pip install dostoevsky
```
```bash
$ pip install natasha
```
```bash
$ pip install pandas
```
```bash
$ pip install sklearn
```
```bash
$ pip install plotly
```
```bash
$ pip install wordcloud
```
# Работа с Lensky
Основным объектом в Lensky является Corpus. Corpus - словарь, состоящий из словарей и функции для их обработки и быстрому доступу к элементам. 

<br/> Corpus : {key1 : {subkey1 : element1, subkey2 : element2...}...}

Инициализировать объект Corpus можно следующим образом:

```python
>>> from Lensky.lensky import *

>>> corpus = Corpus(path_to_folder) # например 'data/corpus_ru'
```

Элементы в папке по адресу path_to_folder должны быть формата .txt и названы по тому же шаблону, что и элементы в data/corpus_ru, то есть - key.subkey.year. 

Также для инициализации можно использовать словарь из словарей следующим образом:

```python
>>> in_dict = {'key1' : {'subkey1' : 'text1', 'subkey2' : 'text2'}, 
          'key2' : {'subkey1' : 'text1'}}
>>> corpus = Corpus(corpus=in_dict)
```

Объект Corpus поддерживает следующие методы:
.get(address) - вернуть элемент корпуса по адресу. Переменная address должна быть списком, первый элемент которого указывает на key, а второй на subkey. Если в списке будет один элемент, то метод вернет словарь, привязанный к key
```python
>>> corpus.get(address) #например [0, 0] или [0]
```
.keys() - вернет все key корпуса в виде списка.
```python
>>> corpus.keys()
```
.values() - вернет все values корпуса в виде списка.
```python
>>> corpus.values()
```
.subkeys() - вернет все subkeys корпуса в виде списка.
```python
>>> corpus.subkeys()
```
.subvalues() - вернет все subvalues элементы корпуса в виде списка.
```python
>>> corpus.subvalues()
```
.booknames() - вернет названия всех книг в корпусе (именно книг, а не элементов в формате key.subkey.year как метод .subvalues()).
```python
>>> corpus.booknames()
```
.apply(func, replace) - применит функцию func к каждому subvalue элементу корпуса. Если replace=False вернет новый объект корпуса, оставив старый без изменений, если replace=True, изменит оригинальный корпус, к которому применен метод. По умолчанию replace=False.
```python
>>> corpus2 = corpus.apply(len, replace=False) # создаст новый объект corpus2 без потери старого, в котором будут содержаться длины каждого элементы subvalue
# ИЛИ
>>> corpus.apply(len, replace=True) # заменит элементы subvalue корпуса на значения их длин
```
.save(path_to_folder) - сохранит корпус по указанному пути в формате data/corpus_ru.
```python
>>> corpus.save(path_to_folder)
```
.tokenize(stop_words, replace) - проведет токенизацию каждого subvalue элемента корпуса. Слова из списка stop_words будут удалены из текстов. По умолчанию список stop_words пуст. Если replace=False вернет новый объект корпуса, оставив старый без изменений, если replace=True, изменит оригинальный корпус, к которому применен метод. По умолчанию replace=False.
```python
>>> corpus2 = corpus.tokenize(stop_words=['я', 'мы', 'ты', 'вы', 'он' , 'она', 'оно', 'они'], replace=False) # создаст новый объект corpus2 без потери старого, в котором будут содержаться токенизированные элементы subvalue
# ИЛИ
>>> corpus.tokenize(stop_words=['я', 'мы', 'ты', 'вы', 'он' , 'она', 'оно', 'они'], replace=True) # заменит элементы subvalue корпуса на их токены
```
.normalize(stop_words, replace) - проведет нормализацию (лемматизацию) каждого subvalue элемента корпуса. Слова из списка stop_words будут удалены из текстов. По умолчанию список stop_words пуст. Если replace=False вернет новый объект корпуса, оставив старый без изменений, если replace=True, изменит оригинальный корпус, к которому применен метод. По умолчанию replace=False. Для этой операции корпус должен быть токенезирован методом .tokenize().
```python
>>> corpus2 = corpus.normalize(stop_words=['я', 'мы', 'ты', 'вы', 'он' , 'она', 'оно', 'они'], replace=False) # создаст новый объект corpus2 без потери старого, в котором будут содержаться лемматизированные элементы subvalue
# ИЛИ
>>> corpus.normalize(stop_words=['я', 'мы', 'ты', 'вы', 'он' , 'она', 'оно', 'они'], replace=True) # заменит элементы subvalue корпуса на их нормальную форму
```
.low_pop_drop(lim, replace) - удалит из текстов слова, доля которых в произведениях меньше, чем lim (рекомендуемое lim<0.001). Если replace=False вернет новый объект корпуса, оставив старый без изменений, если replace=True, изменит оригинальный корпус, к которому применен метод. По умолчанию replace=False. Для этой операции корпус должен быть токенезирован методом .tokenize() и нормализирован методом .normalize().
```python
>>> corpus2 = corpus.low_pop_drop(lim=0.001, replace=False) # создаст новый объект corpus2 без потери старого, в котором будут содержать слова элементов subvalues, прошедшие через фильтр
# ИЛИ
>>> corpus.low_pop_drop(lim=0.001, replace=True) # заменит элементы subvalue корпуса на эти слова
```
.vectorize(vectorizer, replace) - проведет векторизацию каждого subvalue элемента корпуса переданным в vectorizer векторайзером. Если replace=False вернет новый объект корпуса, оставив старый без изменений, если replace=True, изменит оригинальный корпус, к которому применен метод. По умолчанию replace=False.
Применение Count Vectorizer:
```python
>>> from sklearn.feature_extraction.text import CountVectorizer

>>> count = CountVectorizer()

>>> corpus_count = corp.vectorize(count, replace=False)
```
Применение TfidfVectorizer TF:
```python
>>> from sklearn.feature_extraction.text import TfidfVectorizer

>>> tfidf = TfidfVectorizer(use_idf=False)

>>> corpus_tfidf = corp.vectorize(tfidf, replace=False)
```
Применение TfidfVectorizer TF-IDF:
```python
>>> from sklearn.feature_extraction.text import TfidfVectorizer

>>> tfidf = TfidfVectorizer(use_idf=true)

>>> corpus_tfidf = corp.vectorize(tfidf, replace=False)
```
.nearest_neighbours(target, n, metric, model) - найдет n ближайших соседей (по метрике metric, по умолчанию метрика Евклида) элемента, subvalue которого равно target в векторном пространстве порожденном векторайзером. В model можно передать модель для уменьшения размерности. Тогда поиск ближайших соседей будет осуществляться в пространстве уменьшенной размерности. По умолчанию используются расстояния пространства исходной размерности.
```python
>>> corpus.nearest_neighbours(target='Авдеев.Варенька.1852', n=2) 
```








