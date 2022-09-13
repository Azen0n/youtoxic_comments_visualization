from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image

plt.rcParams.update({'font.size': 10})
plt.rcParams['axes.linewidth'] = 0
plt.rcParams['axes.edgecolor'] = '#F4F4F4'


def plot_word_frequency(words: list[str]):
    words_counter = Counter(words)
    word_frequency = dict(sorted(words_counter.items(), key=lambda item: -item[1]))
    plt.barh(list(word_frequency.keys())[:20], list(word_frequency.values())[:20], color='#4285f4')
    plt.title('Word Frequency')
    plt.show()


def plot_wordcloud(words: list[str]):
    mask = np.array(Image.open('mask.png'))
    wordcloud = WordCloud(width=1800,
                          height=1080,
                          random_state=1,
                          background_color='white',
                          colormap='Set2',
                          collocations=False,
                          mask=mask).generate(" ".join(words))
    plt.imshow(wordcloud)
    plt.title('Word Cloud')
    plt.show()
