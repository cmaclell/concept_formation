import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import spacy


en = spacy.load('en_core_web_sm')
stopwords = list(en.Defaults.stop_words)

# [print(i) for i in sorted(stopwords)]

drop_stopwords = False

avging_window = 500


leaf_color = 'red'
basic_color = "green"
best_color = "blue"
word2vec_color = 'black'

cobweb_file = "cobweb_out.csv"
word2vec_file = "word2vec_out_small.csv"

cobweb_normal_data = pd.read_csv(cobweb_file, usecols=['normal_correct', 'normal_prob_word', 'normal_pred_word', 'word'])
cobweb_basic_data = pd.read_csv(cobweb_file, usecols=['basic_correct', 'basic_prob_word', 'basic_pred_word', 'word', 'basic_count', 'n_training_words'])
cobweb_best_data = pd.read_csv(cobweb_file, usecols=['best_correct', 'best_prob_word', 'best_pred_word', 'word', 'best_count', 'n_training_words'])
word2vec_data = pd.read_csv(word2vec_file)

if drop_stopwords:
    cobweb_normal_data.drop(cobweb_normal_data.index[(cobweb_normal_data["word"].apply(lambda x: x in stopwords))], axis=0, inplace=True)
    cobweb_normal_data.index = range(len(cobweb_normal_data.index))
    cobweb_basic_data.drop(cobweb_basic_data.index[(cobweb_basic_data["word"].apply(lambda x: x in stopwords))], axis=0, inplace=True)
    cobweb_basic_data.index = range(len(cobweb_basic_data.index))
    cobweb_best_data.drop(cobweb_best_data.index[(cobweb_best_data["word"].apply(lambda x: x in stopwords))], axis=0, inplace=True)
    cobweb_best_data.index = range(len(cobweb_best_data.index))
    print(len(cobweb_best_data.index))
    word2vec_data.drop(word2vec_data.index[(word2vec_data["word"].apply(lambda x: x in stopwords))], axis=0, inplace=True)
    word2vec_data.index = range(len(word2vec_data.index))

# cobweb_normal_partial_data = pd.read_csv("partial_out_cobweb_normal.csv")
# cobweb_basic_partial_data = pd.read_csv("partial_out_cobweb_basic.csv")
# cobweb_best_partial_data = pd.read_csv("partial_out_cobweb_best.csv")
# word2vec_partial_data = pd.read_csv("partial_out_word2vec.csv")

cobweb_normal_data['normal_percent_correct'] = cobweb_normal_data.normal_correct.rolling(window=avging_window).mean()
cobweb_basic_data['basic_percent_correct'] = cobweb_basic_data.basic_correct.rolling(window=avging_window).mean()
cobweb_best_data['best_percent_correct'] = cobweb_best_data.best_correct.rolling(window=avging_window).mean()
cobweb_normal_data['normal_avg_prob_word'] = cobweb_normal_data.normal_prob_word.rolling(window=avging_window).mean()
cobweb_basic_data['basic_avg_prob_word'] = cobweb_basic_data.basic_prob_word.rolling(window=avging_window).mean()
cobweb_best_data['best_avg_prob_word'] = cobweb_best_data.best_prob_word.rolling(window=avging_window).mean()

cobweb_basic_data['basic_percentage_words'] = (cobweb_basic_data.basic_count / cobweb_basic_data.n_training_words).rolling(window=avging_window).mean()
cobweb_best_data['best_percentage_words'] = (cobweb_best_data.best_count / cobweb_best_data.n_training_words).rolling(window=avging_window).mean()

word2vec_data['percent_correct'] = word2vec_data.correct.rolling(window=avging_window).mean()
word2vec_data['avg_prob_word'] = word2vec_data.prob_word.rolling(window=avging_window).mean()

# cobweb_normal_partial_data['avg_sim'] = cobweb_normal_partial_data.Similarity.rolling(window=avging_window).mean()
# cobweb_basic_partial_data['avg_sim'] = cobweb_basic_partial_data.Similarity.rolling(window=avging_window).mean()
# cobweb_best_partial_data['avg_sim'] = cobweb_best_partial_data.Similarity.rolling(window=avging_window).mean()
# word2vec_partial_data['avg_sim'] = word2vec_partial_data.Similarity.rolling(window=avging_window).mean()

plt.rcParams["figure.figsize"] = [7.00, 7.50]

plt.subplot(3, 1, 1)
plt.plot(cobweb_normal_data.index, cobweb_normal_data.normal_percent_correct, color=leaf_color, label='cobweb leaf')
plt.plot(cobweb_basic_data.index, cobweb_basic_data.basic_percent_correct, color=basic_color, label='cobweb basic')
plt.plot(cobweb_best_data.index, cobweb_best_data.best_percent_correct, color=best_color, label='cobweb best')
plt.plot(word2vec_data.index, word2vec_data.percent_correct, color=word2vec_color, label='word2vec')
plt.title('Rate of correct guesses')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(cobweb_normal_data.index, cobweb_normal_data.normal_avg_prob_word, color=leaf_color, label='cobweb leaf')
plt.plot(cobweb_basic_data.index, cobweb_basic_data.basic_avg_prob_word, color=basic_color, label='cobweb basic')
plt.plot(cobweb_best_data.index, cobweb_best_data.best_avg_prob_word, color=best_color, label='cobweb best')
plt.plot(word2vec_data.index, word2vec_data.avg_prob_word, color=word2vec_color, label='word2vec')
plt.title('Average probability of correct word')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(cobweb_basic_data.index, cobweb_basic_data.basic_percentage_words, color=basic_color, label='cobweb basic')
plt.plot(cobweb_best_data.index, cobweb_best_data.best_percentage_words, color=best_color, label='cobweb best')
plt.title('Percentage of words in predicting node')
plt.legend()

# plt.subplot(3, 1, 3)
# plt.plot(cobweb_normal_partial_data.index, cobweb_normal_partial_data.avg_sim, color=leaf_color, label='cobweb leaf')
# plt.plot(cobweb_basic_partial_data.index, cobweb_basic_partial_data.avg_sim, color=basic_color, label='cobweb basic')
# plt.plot(cobweb_best_partial_data.index, cobweb_best_partial_data.avg_sim, color=best_color, label='cobweb best')
# plt.plot(word2vec_partial_data.index, word2vec_partial_data.avg_sim, color=word2vec_color, label='word2vec')
# plt.title('Average similarity score')
# plt.legend()

plt.tight_layout()
plt.show()
