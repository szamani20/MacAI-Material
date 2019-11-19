## McMaster Artificial Intelligence Society

Materials to be presented by me in the incoming tutorial sessions
by **McMaster AI Society**.

https://docs.google.com/presentation/d/11OcJr7VsOQpm482QAJDgOKNhilkZnNNmeZazkzHE94Q/edit?usp=sharing

**Important note: We cover the Math and Statistics behind every
part of the code that we write as well as explaining the code itself.
It's not a code-only tutorial.**


### Natural Language Processing

##### Basic NLP tasks

In the first tutorial we introduce `Natural Language Processing` or NLP for
short and then go through some basic tasks and methods in NLP using the
`Spacy` library for Python.

Some of the contents to be covered:

* Part of Speech Tagging
* Dependencies and Entities
* Tokenizing text
* Redaction and Sanitization
* Lemmatization
* Similarity calculation
* Stop Words


Here's the link to the notebook:

[https://github.com/szamani20/MacAI-Material/blob/master/nlp/basics_spacy.ipynb](https://github.com/szamani20/MacAI-Material/blob/master/nlp/basics_spacy.ipynb)

---
##### Topic Modeling

For the second tutorial first we cover `Topic Modeling` using
`Latent Dirichlet Allocation` or LDA for short.

We use Amazon's reviews on Fine Food products as our dataset in
this section. Download link:

[https://www.kaggle.com/sdxingaijing/topic-model-lda-algorithm/data](https://www.kaggle.com/sdxingaijing/topic-model-lda-algorithm/data)

And the notebook is here:

[https://github.com/szamani20/MacAI-Material/blob/master/nlp/unsupervised_topic_modeling_lda.ipynb](https://github.com/szamani20/MacAI-Material/blob/master/nlp/unsupervised_topic_modeling_lda.ipynb)

---
##### Text Summarization
For the second part of the second tutorial we cover `Text Summarization`.

`Text Summarization` is somehow related to `Topic Modeling` that we covered in the last
section. In the former we try to find or generate some sentences that can summarize
the whole document and in the latter we try to find which words group together to
form a topic and which topics group together to form a text.

There are two main methods for `Text Summarization`. One is `Extractive` which
generates summary using sentences and words from the text itself without
generating new sentences or new words. The other one is known as `Abstractive`
which generates summary using restructured and new sentences and words.

In this tutorial we only cover `Extractive` text summarization because we haven't
covered some of the material needed for the `Abstractive` method yet.

We summarize the McMaster University Wikipedia page. Feel free to try it on some
other text.

The notebook is here:

[https://github.com/szamani20/MacAI-Material/blob/master/nlp/extractive_text_summarization.ipynb](https://github.com/szamani20/MacAI-Material/blob/master/nlp/extractive_text_summarization.ipynb)

---
##### Sentiment Analysis
For the third tutorial we cover `Sentiment Analysis`.

In the first section of the tutorial we cover `Sentiment Analysis` using
the existing classification algorithms. Given tweets about six US airlines
as our dataset, we try to perform sentiment analysis on the textual
data to analyze whether each tweet is a positive, neutral or negative
comment about the airline.

Some of the contents to be covered:
* Exploratory Data Analysis (EDA)
* TF-IDF text vectorization method
* Naive Bayes Algorithm (for Classification)
* Logistic Regression Algorithm (for Classification)
* Support Vector Machine Algorithm (for Classification)
* Random Forest Algorithm (for Classification)

Here is the link to dataset:

[https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv](https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv)

We perform four supervised classification algorithms in form of a
text classification task to do the sentiment analysis.

Find the notebook here:

[https://github.com/szamani20/MacAI-Material/blob/master/nlp/sentiment_sklearn.ipynb](https://github.com/szamani20/MacAI-Material/blob/master/nlp/sentiment_sklearn.ipynb)

---
##### Deep Learning based Sentiment Analysis with LSTM and CNN
As the second section of the third tutorial we continue focusing
on `Sentiment Analysis` as an application of `NLP` by focusing
on `Deep Learning` methods to fulfill the task.

Some of the contents to be covered:
* Word Embedding
    * Word2Vec
    * GloVe
* Deep Dense Neural Network
* Deep Convolutional Neural Network
* Deep Recurrent Neural Network

To observe probably the best `Sentiment Analysis` model, we start this section of tutorial
by first performing the `Sentiment Analysis` task using a simple `Neural Network`, and then
using `Convolutional Neural Network` and finally using `Long Short Term Memory Network` which
is a type of `Recurrent Neural Networks`.

We finally compare the results from each method and analyze each of them.

We use `IMDB` review dataset which consists of review text and its corresponding sentiment:

[https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

You may also need to download the `GloVe` file that we used in this tutorial here:

[https://www.kaggle.com/terenceliu4444/glove6b100dtxt/data](https://www.kaggle.com/terenceliu4444/glove6b100dtxt/data)

The notebook is here:

[https://github.com/szamani20/MacAI-Material/blob/master/nlp/deep_learning_sentiment_dense_cnn_lstm.ipynb](https://github.com/szamani20/MacAI-Material/blob/master/nlp/deep_learning_sentiment_dense_cnn_lstm.ipynb)

---

I used multiple references to develop the NLP tutorial contents.
A lot of concepts are covered and a lot of references were used to prepare them.
Some of them are listed here:

[https://keras.io/](https://keras.io/)

[https://spacy.io/](https://spacy.io/)

[https://nlp.stanford.edu/IR-book](https://nlp.stanford.edu/IR-book)

[https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)

[https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f](https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f)

[https://blog.keras.io/](https://blog.keras.io/)

[https://www.oreilly.com/library/view/applied-text-analysis/9781491963036/ch04.html](https://www.oreilly.com/library/view/applied-text-analysis/9781491963036/ch04.html)

[https://scikit-learn.org/](https://scikit-learn.org/)

[https://stackabuse.com/](https://stackabuse.com/)

[https://statquest.org/](https://statquest.org/)

[http://karpathy.github.io/2015/05/21/rnn-effectiveness/](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

[http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

[http://colah.github.io/posts/2014-07-Conv-Nets-Modular/](http://colah.github.io/posts/2014-07-Conv-Nets-Modular/)

[http://colah.github.io/posts/2014-07-Understanding-Convolutions/](http://colah.github.io/posts/2014-07-Understanding-Convolutions/)

[https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-latent-dirichlet-allocation-437c81220158](https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-latent-dirichlet-allocation-437c81220158)
