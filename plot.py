from matplotlib import pyplot as plt
import numpy as np

#参数设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (5, 3)

#国家和奖牌数据输入、柱状图宽度设置
clusters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
             '18', '19', '20', '21', '22', '23', '24', '25', '26']
blackspot = np.array([1, 5, 25, 21, 16, 0, 0, 0, 28, 14, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0])
canker = np.array([0, 0, 0, 0, 0, 50, 34, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 1, 0, 0, 0, 10, 0, 0, 0, 0])
huanglong = np.array([7, 35, 13, 3, 3, 0, 0, 0, 0, 1, 38, 4, 3, 4, 0, 6, 5, 0, 14, 0, 5, 0, 0, 0, 0, 5, 4])
healthy = np.array([0, 1, 0, 0, 0, 0, 0, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 7, 0, 0, 0, 0, 0, 0, 0])
melanose = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0])
width = 0.7

#绘图
plt.bar(clusters, blackspot, color='lightgreen', label='Blackspot',
        bottom=canker + huanglong + healthy + melanose, width=width)
plt.bar(clusters, canker, color='deepskyblue', label='Canker', bottom=huanglong + healthy + melanose, width=width)
plt.bar(clusters, huanglong, color='lightsalmon', label='Huanglong', bottom=healthy + melanose, width=width)
plt.bar(clusters, healthy, color='lightgrey', label='Healthy', bottom=melanose, width=width)
plt.bar(clusters, melanose, color='pink', label='Melanose', width=width)

#设置y轴标签，图例和文本值
plt.xlabel('Cluster', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='upper right', prop={'size': 16})
for i in range(len(clusters)):
    max_y = blackspot[i] + canker[i] + huanglong[i] + healthy[i] + melanose[i]
    plt.text(clusters[i], max_y, max_y, va="bottom", ha="center", fontsize=12)

plt.show()
