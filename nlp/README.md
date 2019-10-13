### Natural Language Processing

##### Basic NLP tasks

In this tutorial on NLP, we first start with basic NLP tasks such as `POS tagging`,
`Tokenization`, and `Lemmatization` with `Spacy` library. No dataset is used in
this intro tutorial.

Here's the link to the notebook:

[https://github.com/szamani20/MacAI-Material/blob/master/nlp/basics_spacy.ipynb](https://github.com/szamani20/MacAI-Material/blob/master/nlp/basics_spacy.ipynb)

---
##### Topic Modeling

Then we move forward to cover `Topic Modeling` with the `Latent Dirichlet Allocation`
or simply `LDA` method.

We use Amazon's reviews on Fine Food products as our dataset in this section. Download link:

[https://www.kaggle.com/sdxingaijing/topic-model-lda-algorithm/data](https://www.kaggle.com/sdxingaijing/topic-model-lda-algorithm/data)

And the notebook is here:

[https://github.com/szamani20/MacAI-Material/blob/master/nlp/unsupervised_topic_modeling_lda.ipynb](https://github.com/szamani20/MacAI-Material/blob/master/nlp/unsupervised_topic_modeling_lda.ipynb)

---
##### Sentiment Analysis

Then we cover `Sentiment Analysis` using the existing classification algorithms. Given tweets
about six US airlines as our dataset, we try to perform a sentiment analysis on the text
data to analyze whether each tweet is a positive, neutral or negative comment about the airline.

Here is the link to dataset:

[https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv](https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv)

We perform four supervised classification algorithms in form of a text classification task
to do the sentiment analysis.

Find the notebook here:

[https://github.com/szamani20/MacAI-Material/blob/master/nlp/sentiment_sklearn.ipynb](https://github.com/szamani20/MacAI-Material/blob/master/nlp/sentiment_sklearn.ipynb)

---
##### Deep Learning based Sentiment Analysis with LSTM and CNN

To observe probably the best sentiment analysis model, we start this section of tutorial
by first performing the sentiment analysis task using a simple `Neural Network`, and then
using `Convolutional Neural Network` and finally using `Long Short Term Memory Network` which
is a type of `Recurrent Neural Networks`.

We use IMDB review dataset which consists of review text and its corresponding sentiment:

[https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

You may also need to download the `GloVe` file that we used in this tutorial here:

[https://www.kaggle.com/terenceliu4444/glove6b100dtxt/data](https://www.kaggle.com/terenceliu4444/glove6b100dtxt/data)

The notebook is here:

[https://github.com/szamani20/MacAI-Material/blob/master/nlp/deep_learning_sentiment_dense_cnn_lstm.ipynb](https://github.com/szamani20/MacAI-Material/blob/master/nlp/deep_learning_sentiment_dense_cnn_lstm.ipynb)

---

I used multiple references to develop the NLP tutorial content. Some of them are listed here:

[https://keras.io/](https://keras.io/)

[https://spacy.io/](https://spacy.io/)

[https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f](https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f)

[https://blog.keras.io/](https://blog.keras.io/)

[https://stackabuse.com/](https://stackabuse.com/)

[https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-latent-dirichlet-allocation-437c81220158](https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-latent-dirichlet-allocation-437c81220158)
