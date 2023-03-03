import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


if __name__ == "__main__":

    
    cobweb_data = pd.read_csv("cobweb_rocstories_out.csv")
    word2vec_data = pd.read_csv("word2vec_rocstories_out.csv")
    data = cobweb_data.append(word2vec_data, ignore_index=True)
    print(data)

    data['n_word_blocks'] = data['n_training_words'] % 50

    sns.lineplot(data=data, x="n_word_blocks", y="correct", hue="model")
    # sns.lineplot(data=data, x="n_word_blocks", y="prob_word", hue="model")

    plt.show()
    # plt.savefig("accuracy_by_n.png")
