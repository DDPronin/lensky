# Lensky: Russian fiction literature analysis toolkit


![alt tag](https://github.com/DDPronin/lensky/blob/main/images/Lensky_badge.png)

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
in_dict = {'key1' : {'subkey1' : 'text1', 'subkey2' : 'text2'}, 
          'key2' : {'subkey1' : 'text1'}}
corpus = Corpus(corpus=in_dict)
```

Объект Corpus поддерживает следующие методы:

.get(address) - вернуть элемент корпуса по адресу. Переменная address должна быть списком, первый элемент которого указывает на key, а второй на subkey. Если в списке будет один элемент, то метод вернет словарь, привязанный к key
```python
corpus.get(address) #например [0, 0] или [0]
```

.keys() - вернет все key корпуса в виде списка.
```python
corpus.keys()
```

.values() - вернет все values корпуса в виде списка.
```python
corpus.values()
```

.subkeys() - вернет все subkeys корпуса в виде списка.
```python
corpus.subkeys()
```

.subvalues() - вернет все subvalues элементы корпуса в виде списка.
```python
corpus.subvalues()
```

.booknames() - вернет названия всех книг в корпусе (именно книг, а не элементов в формате key.subkey.year как метод .subvalues()).
```python
corpus.booknames()
```

.apply(func, replace) - применит функцию func к каждому subvalue элементу корпуса. Если replace=False вернет новый объект корпуса, оставив старый без изменений, если replace=True, изменит оригинальный корпус, к которому применен метод. По умолчанию replace=False.
```python
corpus2 = corpus.apply(len, replace=False) # создаст новый объект corpus2 без потери старого, в котором будут содержаться длины каждого элементы subvalue
# ИЛИ
corpus.apply(len, replace=True) # заменит элементы subvalue корпуса на значения их длин
```

.save(path_to_folder) - сохранит корпус по указанному пути в формате data/corpus_ru.
```python
corpus.save(path_to_folder)
```

.tokenize(stop_words, replace) - проведет токенизацию каждого subvalue элемента корпуса. Слова из списка stop_words будут удалены из текстов. По умолчанию список stop_words пуст. Если replace=False вернет новый объект корпуса, оставив старый без изменений, если replace=True, изменит оригинальный корпус, к которому применен метод. По умолчанию replace=False.
```python
corpus2 = corpus.tokenize(stop_words=['я', 'мы', 'ты', 'вы', 'он' , 'она', 'оно', 'они'], replace=False) # создаст новый объект corpus2 без потери старого, в котором будут содержаться токенизированные элементы subvalue
# ИЛИ
corpus.tokenize(stop_words=['я', 'мы', 'ты', 'вы', 'он' , 'она', 'оно', 'они'], replace=True) # заменит элементы subvalue корпуса на их токены
```

.normalize(stop_words, replace) - проведет нормализацию (лемматизацию) каждого subvalue элемента корпуса. Слова из списка stop_words будут удалены из текстов. По умолчанию список stop_words пуст. Если replace=False вернет новый объект корпуса, оставив старый без изменений, если replace=True, изменит оригинальный корпус, к которому применен метод. По умолчанию replace=False. Для этой операции корпус должен быть токенезирован методом .tokenize().
```python
corpus2 = corpus.normalize(stop_words=['я', 'мы', 'ты', 'вы', 'он' , 'она', 'оно', 'они'], replace=False) # создаст новый объект corpus2 без потери старого, в котором будут содержаться лемматизированные элементы subvalue
# ИЛИ
corpus.normalize(stop_words=['я', 'мы', 'ты', 'вы', 'он' , 'она', 'оно', 'они'], replace=True) # заменит элементы subvalue корпуса на их нормальную форму
```

.low_pop_drop(lim, replace) - удалит из текстов слова, доля которых в произведениях меньше, чем lim (рекомендуемое lim<0.001). Если replace=False вернет новый объект корпуса, оставив старый без изменений, если replace=True, изменит оригинальный корпус, к которому применен метод. По умолчанию replace=False. Для этой операции корпус должен быть токенезирован методом .tokenize() и нормализирован методом .normalize().
```python
corpus2 = corpus.low_pop_drop(lim=0.001, replace=False) # создаст новый объект corpus2 без потери старого, в котором будут содержать слова элементов subvalues, прошедшие через фильтр
# ИЛИ
corpus.low_pop_drop(lim=0.001, replace=True) # заменит элементы subvalue корпуса на эти слова
```

.vectorize(vectorizer, replace) - проведет векторизацию каждого subvalue элемента корпуса переданным в vectorizer векторайзером. Если replace=False вернет новый объект корпуса, оставив старый без изменений, если replace=True, изменит оригинальный корпус, к которому применен метод. По умолчанию replace=False.

Применение Count Vectorizer:
```python
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()

corpus_count = corp.vectorize(count, replace=False)
```
Применение TfidfVectorizer TF:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(use_idf=False)

corpus_tfidf = corp.vectorize(tfidf, replace=False)
```
Применение TfidfVectorizer TF-IDF:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(use_idf=True)

corpus_tfidf = corp.vectorize(tfidf, replace=False)
```

.nearest_neighbours(target, n, metric, model) - найдет n ближайших соседей (по метрике metric, по умолчанию метрика Евклида) элемента, subvalue которого равно target в векторном пространстве, порожденном векторайзером. В model можно передать модель для уменьшения размерности. Тогда поиск ближайших соседей будет осуществляться в пространстве уменьшенной размерности. По умолчанию используются расстояния пространства исходной размерности.

```python
corpus_tfidf.nearest_neighbours(target='Авдеев.Варенька.1852', n=2) 
```

После векторизации .vectorize каждый элемент subvalue корпуса превращается в Vector - объект, обладающий своими методами:

.get_principal_words(num) - получить num слов, отвечающих за самые значимые компоненты вектора векторизованного корпуса. Если не инициализировать num, то будут выведены все слова в порядке уменьшения величины компоненты. 

```python
vector = corpus_tfidf.get([0, 0])
vector.get_principal_words(num=10)
```

get_principal_numbers(num) - получить num значений компонент вектора векторизованного корпуса. Если не инициализировать num, то будут выведены компоненты в порядке уменьшения.

```python
vector = corpus_tfidf.get([0, 0])
vector.get_principal_numbers(num=10)
```

<br/> Функции для работы с корпусом:

plot_by_keys(corpus, model, colors, labels, point_size, alpha, title, legend_title, grid, figsize, dpi) - построение графика по ключам keys. На графике будут выделены цветами keys. Аргументы: corpus - векторизованный корпус (объект Corpus), model - предобученная модель для уменьшения размерности, colors - цвета точек (заданы по умолчанию, но можно изменить), labels - подписи, отображаемые в легенде (заданы по умолчанию, но можно изменить), point_size - размер точки (задан по умолчанию, но можно изменить), alpha - прозрачность точки (задана пор умолчанию, но можно изменить), title - подпись графика, legend_title - подпись легенды графика, grid - сетка (вкл./выкл True/False), figsize, dpi - настройки фигуры графика, соотвествуют настройкам из matplotlib

```python
import umap

def cent(year): # функция для определения века написания
    if year < 1801:
        return 18
    if year >= 1801 and year < 1901:
        return 19
    if year >= 1901 and year < 2001:
        return 20
    if year >= 2001:
        return 21
        
cent_tfidf = Corpus(corpus={18 : {}, 19 : {}, 20 : {}, 21 : {}}) # создадим новый корпус, где ключами будут не авторы, а века написания книгb

for key in corp.keys(): # определим века написания
    for subkey in corpus_tfidf[key].keys():
        year = int(subkey.split('.')[-1])
        cent_tfidf[cent(year)][subkey] = author_tfidf[key][subkey]
       

mapper = umap.UMAP() # создадим модель UMAP для уменьшения размерности
model_fit(cent_tfidf, mapper) # обучим ее на данных

plot_by_keys(cent_tfidf, mapper, figsize=(10, 10), dpi=150, title='Отображение UMAP произведений по векам', legend_title='Век') # построим график
```

![alt tag](https://github.com/DDPronin/lensky/blob/main/images/UMAP_Gogol.png)

plot_by_subkeys(corpus, target, model, colors, labels, point_size, alpha, title, legend_title, grid, figsize, dpi) - построение графика по объекту key и его subkeys "на фоне" всего корпуса. Аргументы: corpus - векторизованный корпус (объект Corpus), target - выделяемый key, model - предобученная модель для уменьшения размерности, colors - цвета точек (заданы по умолчанию, но можно изменить), labels - подписи, отображаемые в легенде (заданы по умолчанию, но можно изменить), point_size - размер точки (задан по умолчанию, но можно изменить), alpha - прозрачность точки (задана пор умолчанию, но можно изменить), title - подпись графика, legend_title - подпись легенды графика, grid - сетка (вкл./выкл True/False), figsize, dpi - настройки фигуры графика, соотвествуют настройкам из matplotlib

```python
import umap

mapper = umap.UMAP() # создадим модель UMAP для уменьшения размерности
model_fit(corpus_tfidf, mapper) # обучим ее на данных

plot_by_subkeys(author_tfidf, 'Гоголь', mapper, figsize=(10, 10), dpi=250, point_size=14, legend_title='Произведения автора') # построение графика
```

plot_interactive(corpus, model, data_key, data_subkey) - постройка анимированного графика по корпусу. Аргументы: corpus - векторизованный корпус (объект Corpus), model - предобученная модель для уменьшения размерности, data_key, data_subkey - подписи для легенды.

```python
import umap

def cent(year): # функция для определения века написания
    if year < 1801:
        return 18
    if year >= 1801 and year < 1901:
        return 19
    if year >= 1901 and year < 2001:
        return 20
    if year >= 2001:
        return 21
        
cent_tfidf = Corpus(corpus={18 : {}, 19 : {}, 20 : {}, 21 : {}}) # создадим новый корпус, где ключами будут не авторы, а века написания книгb

for key in corp.keys(): # определим века написания
    for subkey in corpus_tfidf[key].keys():
        year = int(subkey.split('.')[-1])
        cent_tfidf[cent(year)][subkey] = author_tfidf[key][subkey]
       

mapper = umap.UMAP() # создадим модель UMAP для уменьшения размерности
model_fit(cent_tfidf, mapper) # обучим ее на данных

plot_interactive(cent_tfidf, mapper) # построим график
```

<br/> Функции для работы с векторами:

vsum([vector1, vector2...]) - найдет векторную сумму списка [vector1, vector2...]

```python
vector1 = corpus_tfidf.get([0, 0])
vector1 = corpus_tfidf.get([0, 1])
vsum([vector1, vector2])
```

vmean([vector1, vector2...]) - найдет среднее векторов списка [vector1, vector2...]

```python
vector1 = corpus_tfidf.get([0, 0])
vector1 = corpus_tfidf.get([0, 1])
vmean([vector1, vector2])
```

vdif(vector1, vector2) - вычтет из vector1 vector2 (vector1-vector2)

```python
vector1 = corpus_tfidf.get([0, 0])
vector1 = corpus_tfidf.get([0, 1])
vdif(vector1, vector2)
```

<br/> Функции для работы с текстом (строками str):

sentiment_plot(book, window, plot_type, title, zero_level, grid, figsize, dpi) - функция для построения графика анализа тональности текста. Аргументы: book - строка str, содержащая текст книги, window - размер окна (см. прикрепленную статью), plot_type - можно задать список list с элементами из множества ['neutral', 'positive', 'negative', 'speech'], тогда функция изобразит график только этих типов, также можно задать аргумент строкой 'sentiment', 'pos-neg' или 'comb', тогда функция нарисует специализированный график, изобразив уровень сентиментальности, общей тональности или все сразу. Примеры: plot_type = ['positive', 'negative], plot_type='pos-neg' и тд. Аргумент title - подпись с графику, zero_level - нулевой уровень (вкл./выкл True/False), grid - сетка (вкл./выкл True/False), figsize, dpi - настройки фигуры графика, соотвествуют настройкам из matplotlib

```python
from Lensky.lensky import *

corpus = Corpus('data/corpus_ru')

text = corp['Лермонтов']['Лермонтов.Герой_нашего_времени.1840']

sentiment_plot(text, plot_type='sentiment', window=200, title='', grid=True, zero_level=False)
```

morph_plot(book, window, plot_type, title, grid, figsize, dpi) - функция для построения графика долей частей речи в тексте. Аргументы: book - строка str, содержащая текст книги, window - размер окна (см. прикрепленную статью), plot_type - можно задать список list с элементами из множества ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']. Примеры: plot_type = ['ADJ', 'PART] - будут построены графики соотвествующих частей речи. Аргумент grid - сетка (вкл./выкл True/False), figsize, dpi - настройки фигуры графика, соотвествуют настройкам из matplotlib

```python
from Lensky.lensky import *

corpus = Corpus('data/corpus_ru')

text = corp['Лермонтов']['Лермонтов.Герой_нашего_времени.1840']

morph_plot(text, plot_type=['VERB'], window=500, title='Глаголы в произведении "Герой нашего времени"', grid=True)
```

plot_wordcloud(book, stopwords) - строит облако слов произведения. book - строка str, содержащая текст книги. Слова из списка stop_words будут удалены из текста.

```python
from Lensky.lensky import *

corpus = Corpus('data/corpus_ru_normalized')

text = corp['Лермонтов']['Лермонтов.Герой_нашего_времени.1840']

plot_wordcloud(text, stopwords=stopwords)
```









