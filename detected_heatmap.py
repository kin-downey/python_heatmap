import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.cm as cm



def generate_random_data(data_length: int):
    """縦960, 横1280のうちランダムに1000点を選択し(x, y)の形式で格納

    Args:
        data_length (int): データの個数
    """
    x = np.random.randint(0, 1280, data_length)
    y = np.random.randint(0, 960, data_length)
    return x, y


def validate_data_length(x: np.array, y: np.array):
    """データの個数を確認（xとyの長さが一致していることを確かめる）

    Args:
        x (np.array): x座標の配列
        y (np.array): y座標の配列
    """
    if len(x) != len(y):
        raise ValueError('xとyのデータ数が異なります．')
    

def generate_heatmap(title: str, x_label: str, y_label: str, x: np.array, y: np.array):
    """ヒートマップ（グリッドタイプ）作成し.pngで保存

    Args:
        title (str): グラフのタイトル（.pngに表示される）
        x_label (str): 横軸のラベル
        y_label (str): 縦軸のラベル
        x (np.array): x座標の配列
        y (np.array): y座標の配列
    """
    validate_data_length(x, y)
    fig, ax = plt.subplots() # 返り値がタプルのため，figを明示的に指定している
    sns.heatmap(np.histogram2d(x, y, bins=[128, 96])[0], cmap='Reds')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.subplots_adjust(top=0.9)
    plt.suptitle(title)
    plt.savefig('detected_heatmap.png')
    plt.show()


def generate_kde_heatmap(title: str, x_label: str, y_label: str, x: np.array, y: np.array):
    """ヒートマップ（なめらか）を作成.pngで保存

    Args:
        title (str): グラフのタイトル（.pngに表示される）
        x_label (str): x軸のラベル
        y_label (str): y軸のラベル
        x (np.array): x座標の配列
        y (np.array): y座標の配列
    """
    validate_data_length(x, y)
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
    ax.invert_yaxis()
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
    # x, y = generate_random_data()
    x = np.random.randint(0, 100, 10)
    y = np.random.randint(0, 100, 10)
    # generate_heatmap(title, x_label, y_label, x, y)
    generate_kde_heatmap(title, x_label, y_label, x, y)