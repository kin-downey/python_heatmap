import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.cm as cm



# 縦960, 横1280のうち，ランダムに1000点を選択し，(x, y)の形式で格納
def generate_random_data():
    x = np.random.randint(0, 1280, 1000)
    y = np.random.randint(0, 960, 1000)
    return x, y


# ヒートマップ（グリッドタイプ）の作成，pngで保存
def generate_heatmap(title, x_label, y_label, x, y):
    # 返り値がタプルのため，figを明示的に指定している
    fig, ax = plt.subplots()
    sns.heatmap(np.histogram2d(x, y, bins=[128, 96])[0], cmap='Reds')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.subplots_adjust(top=0.9)
    plt.suptitle(title)
    plt.savefig('detected_heatmap.png')
    plt.show()


# ヒートマップ（なめらか）の作成，pngで保存
def generate_kde_heatmap(title, x_label, y_label, x, y):
    # ポイントをなめらかにするための設定
    xx,yy = np.mgrid[0:1280:1,0:960:1]
    positions = np.vstack([xx.ravel(),yy.ravel()])
    value = np.vstack([x,y])
    # bw_methodはなめらかさの度合いを指定するパラメータ
    kernel = gaussian_kde(value, bw_method=0.1)
    f = np.reshape(kernel(positions).T, xx.shape)
    # ヒートマップを生成する
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contourf(xx,yy,f, cmap=cm.jet)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.savefig(f'{title}.png')
    plt.show()


if __name__ == '__main__':
    title = 'Detected Heatmap'
    # 軸のラベルはアルファベットのみ指定可能
    x_label = 'X'
    y_label = 'Y'
    x, y = generate_random_data()
    # generate_heatmap(title, x_label, y_label, x, y)
    generate_kde_heatmap(title, x_label, y_label, x, y)