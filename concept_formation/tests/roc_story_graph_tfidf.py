import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import spacy


en = spacy.load('en_core_web_sm')
stopwords = list(en.Defaults.stop_words)

# [print(i) for i in sorted(stopwords)]

drop_stopwords = False

avging_window = 1500


leaf_color = 'red'
basic_color = "green"
best_color = "blue"
word2vec_color = 'black'

cobweb_file = "cobweb_out_unweighted.csv"
other_file = "cobweb_out_weighted.csv"

cobweb_normal_data = pd.read_csv(cobweb_file, usecols=['normal_correct', 'normal_prob_word', 'normal_pred_word', 'word'])
cobweb_basic_data = pd.read_csv(cobweb_file, usecols=['basic_correct', 'basic_prob_word', 'basic_pred_word', 'word', 'basic_count', 'n_training_words'])
cobweb_best_data = pd.read_csv(cobweb_file, usecols=['best_correct', 'best_prob_word', 'best_pred_word', 'word', 'best_count', 'n_training_words'])

other_normal_data = pd.read_csv(other_file, usecols=['normal_correct', 'normal_prob_word', 'normal_pred_word', 'word'])
other_basic_data = pd.read_csv(other_file, usecols=['basic_correct', 'basic_prob_word', 'basic_pred_word', 'word', 'basic_count', 'n_training_words'])
other_best_data = pd.read_csv(other_file, usecols=['best_correct', 'best_prob_word', 'best_pred_word', 'word', 'best_count', 'n_training_words'])


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

other_normal_data['normal_percent_correct'] = other_normal_data.normal_correct.rolling(window=avging_window).mean()
other_basic_data['basic_percent_correct'] = other_basic_data.basic_correct.rolling(window=avging_window).mean()
other_best_data['best_percent_correct'] = other_best_data.best_correct.rolling(window=avging_window).mean()
other_normal_data['normal_avg_prob_word'] = other_normal_data.normal_prob_word.rolling(window=avging_window).mean()
other_basic_data['basic_avg_prob_word'] = other_basic_data.basic_prob_word.rolling(window=avging_window).mean()
other_best_data['best_avg_prob_word'] = other_best_data.best_prob_word.rolling(window=avging_window).mean()

cobweb_basic_data['basic_percentage_words'] = (cobweb_basic_data.basic_count / cobweb_basic_data.n_training_words).rolling(window=avging_window).mean()
cobweb_best_data['best_percentage_words'] = (cobweb_best_data.best_count / cobweb_best_data.n_training_words).rolling(window=avging_window).mean()
other_basic_data['basic_percentage_words'] = (other_basic_data.basic_count / other_basic_data.n_training_words).rolling(window=avging_window).mean()
other_best_data['best_percentage_words'] = (other_best_data.best_count / other_best_data.n_training_words).rolling(window=avging_window).mean()

# word2vec_data['percent_correct'] = word2vec_data.correct.rolling(window=avging_window).mean()
# word2vec_data['avg_prob_word'] = word2vec_data.prob_word.rolling(window=avging_window).mean()

# cobweb_normal_partial_data['avg_sim'] = cobweb_normal_partial_data.Similarity.rolling(window=avging_window).mean()
# cobweb_basic_partial_data['avg_sim'] = cobweb_basic_partial_data.Similarity.rolling(window=avging_window).mean()
# cobweb_best_partial_data['avg_sim'] = cobweb_best_partial_data.Similarity.rolling(window=avging_window).mean()
# word2vec_partial_data['avg_sim'] = word2vec_partial_data.Similarity.rolling(window=avging_window).mean()

plt.rcParams["figure.figsize"] = [7.00, 7.50]

plt.subplot(3, 1, 1)
plt.plot(cobweb_normal_data.index, cobweb_normal_data.normal_percent_correct, color=leaf_color, label='cobweb leaf')
# plt.plot(cobweb_basic_data.index, cobweb_basic_data.basic_percent_correct, color=basic_color, label='cobweb basic')
plt.plot(cobweb_best_data.index, cobweb_best_data.best_percent_correct, color=best_color, label='cobweb best')
plt.plot(other_normal_data.index, other_normal_data.normal_percent_correct, color='magenta', label='cobweb leaf unweighted')
# plt.plot(other_basic_data.index, other_basic_data.basic_percent_correct, color='magenta', label='cobweb basic II')
plt.plot(other_best_data.index, other_best_data.best_percent_correct, color='cyan', label='cobweb best unweighted')
plt.title('Rate of correct guesses')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(cobweb_normal_data.index, cobweb_normal_data.normal_avg_prob_word, color=leaf_color, label='cobweb leaf')
# plt.plot(cobweb_basic_data.index, cobweb_basic_data.basic_avg_prob_word, color=basic_color, label='cobweb basic')
plt.plot(cobweb_best_data.index, cobweb_best_data.best_avg_prob_word, color=best_color, label='cobweb best')
plt.plot(cobweb_normal_data.index, other_normal_data.normal_avg_prob_word, color='magenta', label='cobweb leaf unweighted')
# plt.plot(cobweb_basic_data.index, other_basic_data.basic_avg_prob_word, color='yellow', label='cobweb basic unweighted')
plt.plot(cobweb_best_data.index, other_best_data.best_avg_prob_word, color='cyan', label='cobweb best unweighted')
# plt.plot(word2vec_data.index, word2vec_data.avg_prob_word, color=word2vec_color, label='word2vec')
plt.title('Average probability of correct word')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(cobweb_basic_data.index, cobweb_basic_data.basic_percentage_words, color=basic_color, label='cobweb basic')
plt.plot(cobweb_best_data.index, cobweb_best_data.best_percentage_words, color=best_color, label='cobweb best')
plt.plot(other_basic_data.index, other_basic_data.basic_percentage_words, color='yellow', label='cobweb basic unweighted')
plt.plot(other_best_data.index, other_best_data.best_percentage_words, color='cyan', label='cobweb best unweighted')
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
