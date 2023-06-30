import pandas as pd
from matplotlib import pyplot as plt

cobweb_data = pd.read_csv("cobweb_roc_story_out.csv")
word2vec_data = pd.read_csv("word2vec_roc_story_out.csv")

cobweb_data['percent_correct'] = cobweb_data.correct.rolling(window=500).mean()
cobweb_data['avg_prob_word'] = cobweb_data.prob_word.rolling(window=500).mean()

word2vec_data['percent_correct'] = word2vec_data.correct.rolling(window=500).mean()
word2vec_data['avg_prob_word'] = word2vec_data.prob_word.rolling(window=500).mean()


plt.rcParams["figure.figsize"] = [7.00, 7.50]

plt.subplot(2, 1, 1)
plt.plot(cobweb_data.index, cobweb_data.percent_correct, color='blue', label='cobweb')
plt.plot(word2vec_data.index, word2vec_data.percent_correct, color='red', label='word2vec')
plt.title('Rate of correct guesses')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(cobweb_data.index, cobweb_data.avg_prob_word, color='blue', label='cobweb')
plt.plot(word2vec_data.index, word2vec_data.avg_prob_word, color='red', label='word2vec')
plt.title('Average probability of correct word')
plt.legend()

plt.tight_layout()
plt.show()

# fig1, ax1 = plt.subplots(figsize=(7, 5))
# ax1.plot(cobweb_data.index, cobweb_data.percent_correct, color='blue', label='cobweb')
# ax1.plot(word2vec_data.index, word2vec_data.percent_correct, color='red', label='word2vec')
# ax1.set_title('Rate of correct guesses')
# ax1.legend()

# fig2, ax2 = plt.subplots(figsize=(7, 5))
# ax2.plot(cobweb_data.index, cobweb_data.avg_prob_word, color='blue', label='cobweb')
# ax2.plot(word2vec_data.index, word2vec_data.avg_prob_word, color='red', label='word2vec')
# ax2.set_title('Probability of correct word')
# ax2.legend()

# plt.rcParams["figure.figsize"] = [7.00, 3.50]
# plt.rcParams["figure.autolayout"] = True

# plt.plot(cobweb_data.index, cobweb_data.percent_correct, color='blue')
# plt.plot(word2vec_data.index, word2vec_data.percent_correct, color='red')
