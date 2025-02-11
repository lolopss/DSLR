from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv('dataset_train.csv')

key_names = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', \
    'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', \
    'Transfiguration', 'Potions', 'Care of Magical Creatures', \
    'Charms', 'Flying']

oui = list(combinations(key_names, 2))


colors = {'Gryffindor': 'red', 'Hufflepuff': 'yellow', 'Ravenclaw': 'blue', 'Slytherin': 'green'}
sns.pairplot(data, hue='Hogwarts House', diag_kind='kde', vars=key_names, palette=colors, plot_kws={'alpha':0.6, 's':5, 'edgecolor':'k', 'linewidth':0.5})
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.4, hspace=0.4)
plt.show()

houses = data['Hogwarts House']
divination = data['Divination']

plt.scatter(divination, houses, c=houses.map(colors), alpha=0.5)
plt.title('Scatter Plot of Divination vs Hogwarts House')
plt.xlabel('Divination')
plt.ylabel('Hogwarts House')
plt.show()