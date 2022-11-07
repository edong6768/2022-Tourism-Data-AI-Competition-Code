import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import font_manager, rc

font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

# df = pd.read_csv('./open/train.csv', sep=',', encoding='utf-8')
df = pd.read_csv('./open/aug.csv', sep='\t', encoding='utf-8')

classes = list(set(df.cat3))
cat3 = Counter(df.cat3.tolist())
cat3_l = [*zip(*cat3.most_common())]
print(cat3['한식'])
print([i for i in cat3.most_common() if i[1]==min(cat3_l[1])])
clss = [i[0] for i in cat3.most_common() if i[1]==min(cat3_l[1])]

for c in clss:
    print(*[k for j in df[df.cat3==c].iloc for k in j.overview.split('.')], sep='\n')
    print('\n########################')

# plt.barh(x, cat3_l)
# plt.yticks(x, classes, fontsize=8)

plt.pie(cat3_l[1], labels=cat3_l[0], autopct='%.1f%%', startangle=260, counterclock=False)
plt.show()
