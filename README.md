# General Conversation Challenge


This repo implements retrieval-based dialogue system in Tensorflow.  
This is a solution for ML track of Yandex Algorithm 2018 (https://contest.yandex.com/algorithm2018/). 
The goal of the challenge was to build a system that can rank answers for a dialogue system based on labeled Hollywood movies scripts.

## Task

Given a dialog (from 1 to 3 utterances) and up to 5 possible answers, rank answers based on labeled replies in train dataset.


## Data preprocessing

Data in the assignment is more or less clean. However, it does contain some spelling mistakes and encoding bugs (for example, 'ьl' instead of 'ы'). Because of that, I do some prerocessing during transferring sentence words to indices. 
I make words lowercase as well for simplicity. If context contains more than one utterances, I concatenate them to one long paragraph.


# Model


Main idea is to bring contextual representations of the previous conversation and an approptiate answer closer.

Brief overview:
1) Words -> word vectors using fasttext
2) Word vectors to contextual representation for each word using bi-LSTM layer
3) Dense layer
4) Squeeze bi-LSTM layer outputs for the context and the reply by taking the average vector along the horizontal axis. Resulting size (300, 1)
5) Compute Euclidean distance between two vectors.
6) Minimize MSE between computed distance and value * confidence:
multiplication of class_value (0.001 for 'bad', 1 for 'neutral', 2 for 'good') and confidence (handcrafted number from 0 to 1)

The main disadvantage of such workflow is the step 4 - while squeezing we lost a lot of infomation.
I tried different approaches:
1) Make the matrices squared (multiplication by transposed matrix itself). Averaging approach showed better results.
2) Distance correlation for matrices with different sizes. Did not succeeded in implementing it in tensorflow.
3) Try to convolve matrices to create smaller representations of the same size for the context and the reply and then compute the distance. Didn't succeed in that.


### Word vectors

We encode each word and symbol in the sentence to fasttext vector. I used fasttext vectors pretrained on Wikipedia corpus. 
Words that are not in the corpus are changed for "$UNK$" and all numbers are changed for "$NUM$".


### BiLSTM for contextual Representation

We use bidirectional LSTM to make a use of words surrounding the current word. Output of this layer is a matrix with shape (number of words; lenght of embedding vector) where lenght of embedding vector is equal to 300.


### Split up bad and good replies.

Now we have two matrices: one for context and one for reply. For each reply we know the label ('bad', 'neural' or 'good'). I encoded them with the values: 0.001, 1, 2.
We also know the confidence of the accessors - value from 0 to 1.
Let's set a regression task: try to train algorithm in a way that some sort of distance between context matrix and reply matrix is equal for value * confidence.

The main concern is how to compute distance between two matrices with same width (300) and non-equal lenghts.

The easiest approach is to compute the average vector along the horizontal axis, so matrix (n, 300) can be squeezed into the vector (1, 300)

For such vectors we can compute Euclidean, cosine or whatever distance and try to make it as close as possible to value * confidence.




# Implementation


1) Clone the repo:

```
git clone https://github.com/owoshch/yandex-dialog-challenge.git
```
2) Create the virtual environment:
Note: I use Anaconda for managing environments. However, you can use your favorite approach.

```
conda create -n dialog-env python=3.5.2
source activate dialog-env
```
3) Install requirements: 

```
cd yandex-dialog-challenge
pip install -r requirements.txt
```

4) Install FastText and download pretrained word vectors:

```
make fasttext
```


5) Download pretrained word vectors:

```
make corpus
```


6) Download contest dataset:

```
make dataset
```

7) Prerocess data and train the algorithm:
Note: This step will require significant amount of time. You can skip it and download preprocessed data and pretrained weights.

```
make run
```


8) Optional. Donwload preprocessed data.

```
make data
```


9) Optional. Download pretrained weights.




Russian version of the problem statement:

## Описание задачи
Создание диалоговой системы, которая может разговаривать с пользователем на любые темы, производя впечатление умного и интересного собеседника – одна из самых сложных и важных задач искусственного интеллекта в наши дни. Одна из основных подзадач в создании такой системы – генерация или подбор реплик, подходящих по смыслу для данного момента разговора и способных заинтересовать пользователя в продолжении беседы [1].

Существует два основных подхода к решению этой подзадачи – генеративный [4] и поисковый [3]. Подходы первого типа основаны на построении сложной языковой модели, генерирующей реплику на основании контекста разговора, слово за словом. Подходы второго типа предполагают, что «все, что можно было сказать, уже сказано до нас» и, вместо генерации реплики, ищут наиболее подходящую реплику в большой коллекции существующих реплик. Современные подходы обоих типов, как правило, опираются на глубокие нейронные сети разнообразных архитектур.

Наше соревнование сфокусировано на улучшении второго типа подходов – поисковых (на нем, в частности, основана Алиса, диалоговый помощник, разработанный компанией Яндекс [6]). Заметим, что это не значит, что поисковые методы не могут быть улучшены генеративными – такие гибридные подходы существуют тоже [5]. Мы хотели бы, чтобы участники не упустили возможность попробовать и их.

Мы хотим, чтобы участники нашего соревнования почувствовали себя ближе к созданию настоящего диалогового ассистента, способного увлекательно разговаривать с миллионами пользователей, как это ежедневно делает Алиса.

Для этого мы использовали публичную базу диалогов и краудсорсинговую платформу Яндекс.Толока, чтобы собрать датасет, похожий на тот, который используется для обучения Алисы.

## Описание данных

Каждый из файлов субтитров в датасете OpenSubtitles [2], который мы использовали в качестве источника реплик и разговоров, содержит упорядоченный набор реплик. В большинстве случаев, каждая реплика – это ответ на предыдущую, в разговоре между двумя персонажами фильма. Мы случайно выбрали эпизоды этих разговоров в качестве наших тренировочных и тестовых примеров.

Каждый эпизод состоит из двух частей – контекста (Context) и финальной реплики (Reply). Например,

context_2: Персонаж A говорит реплику 
context_1: Персонаж B отвечает на нее 
context_0: Персонаж А произносит вторую реплику 
reply: Персонаж B отвечает на вторую реплику 
Контекстная часть может состоять из трех реплик (как в примере) – в 50% случаев, двух – в 25%, и одного – в оставшихся 25% случаев. Финальная реплика (Reply) всегда завершает любой эпизод, то есть следует за контекстом (Context). Задача участников – найти наиболее подходящую и интересную реплику для данного контекста среди предложенных кандидатов (числом до 6), случайно выбранных из топа кандидатов, возвращенных бейзлайном высокого качества, натренированным командой Алисы (который, в свою очередь, отобрал кандидатов среди всех возможных реплик OpenSubtitles).

Все реплики-кандидаты размечены асессорами на сервисе Яндекс.Толока с помощью следующей инструкции для разметки:

Good (2): реплика уместна (имеет смысл для данного контекста) и интересна (нетривиальна, специфична именно для данного контекста, мотивирует продолжать разговор)
Neutral (1): реплика уместна (имеет смысл для данного контекста), но не интересна (тривиальна, не специфична для данного контекста и скорее подталкивает пользователя закончить разговор)
Bad (0): реплика не имеет никакого смысла в данном контексте
Каждая метка в тренировочной части датасета (и только в ней), сопровождается также уверенностью (confidence) – числом в интервале от 0 до 1 – которое показывает насколько уверенными в своей разметке были асессоры с Толоки, совместно предложившие данную метку. Мы хотим обратить особое внимание участников на эту информацию, она может быть очень полезна при обучении их моделей.

Мы хотим особо отметить, что все участники имеют право скачать датасет OpenSubtitles [2], который использовался для подготовки датасета и применять его для тренировки своих моделей по своему усмотрению.

## Формат данных 

Каждая строка в тренировочной части датасета представлена в следующем формате:

#context_id,context_2,context_1,context_0,reply_id,reply,label,confidence

context_id – идентификатор эпизода
context_2,context_1,context_0 – текст реплик, предшествующих финальной (может состоять из трех частей)
reply_id – идентификатор реплики-кандидата
reply – текст реплики-кандидата
label – метка реплики-кандидата (good, neutral или bad)
confidence - уверенность в метке реплики-кандидата (число от 0 до 1)
Каждая строка в тестовой части датасета представлена в следующем формате (по аналогии с тренировочной, но без информации о метках):

#context_id,context_2,context_1,context_0,reply_id,reply

Все строки в файле, который присылают участники должны быть организованы следующим образом:

#context_id, reply_id

где все context_id должны быть отсортированы на уровне посылаемого файла в возрастающем порядке
и все reply_id должны быть в порядке ранжирования реплик (то есть, в порядке убывания их скоров), который возвратила ваша система для данного context_id
context_id и reply_id должны быть отделены либо символом пробела либо tab
каждый файл-решение должен содержать то же число строк, что и файл с тестовыми данными


## Метрика

Задача участников – возвратить ранжирование реплик-кандидатов представленных в порядке убывания скоров, выданных моделями участников. Метрика для оценивания этих ранжирований – NDCG. Больше информации о метрике можно найти на вики-странице https://en.wikipedia.org/wiki/Discounted_cumulative_gain - обратите внимание, что мы используем первый из двух вариантов DCG, представленных на странице.

IDCG – это максимально возможное значение метрики DCG для данного набора кандидатов, оно измеряется после ранжирования кандидатов в порядке убывания значений их меток (не предсказанных скоров).

reli принимает три возможных значения - 2, 1 и 0 - для меток good, neutral и bad соответственно.

Особо отмечаем, что информация об уверенности меток (доступная только для тренировочных данных) никак не учитывается в метрике.

Скор участников отображаемый в контесте - это среднее NDCG для всех context_id тестовых данных, умноженное на 100 000.


Links:
[1] A Survey on Dialogue Systems: Recent Advances and New Frontiers. KDD 2017 http://www.kdd.org/exploration_files/19-2-Article3.pdf \
[2] OpenSubtitles http://opus.nlpl.eu/download.php?f=OpenSubtitles2016/en-ru.txt.zip \
[3] https://github.com/faneshion/MatchZoo \
[4] https://github.com/julianser/hed-dlg-truncated \
[5] Two are Better than One: An Ensemble of Retrieval- and Generation-Based Dialog Systems, 2016 https://arxiv.org/pdf/1610.07149.pdf \
[6] https://alice.yandex.ru

Other useful links:  
https://habr.com/company/yandex/blog/349172/  
https://www.youtube.com/watch?v=m4yxsBMBgtM  

