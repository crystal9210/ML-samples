import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA # pca:主成分分析をするためのクラス
# ー＞PCA:多次元データを低次元空間に射影することで、データの次元削減を行う統計手法、より統計的に意味がある次元のみを残し分析に利用することで効果的なデータ分析や加工処理が可能となる

# TODO:ロジスティック回帰の各種理論の再学習

# データセットのロード
iris = load_iris()
X = iris.data
y = iris.target

# PCAで2次元に次元削減
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X) # X_reduced:次元削減されたデータ

# データを訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# モデルの学習
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# 精度の測定
# 今回はラベル付きの教師あり学習ー＞各データのX_testに対して正解；y_testが与えられているため、
# 学習データで学習させた学習器の性能を正解値と予測値を比較することで測定可能
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

# 分類境界のプロット
x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1 # 次元削減されたデータの一つ目の主成分の最小値、最大値をそれぞれ取得し、空によりそれぞれ下限、上限を計算ー＞指定する
y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
# meshgrid:2つの1次元配列から2次元の格子点を生成,arange:第一引数から第二引数までの数値に対して第3引数刻みで格子点の要素を生成
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# ここで予測に使用するのは次元削減後の2次元データ
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, s=20, edgecolor='k')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Logistic Regression Classification Boundaries')
plt.show()
# MEMO - 出力される各数値の属性
# precision（適合率:各クラスに対して、予測された正のデータポイントのうち、実際に正であった割合
# recall（再現率:各クラスに実際に属するデータポイントのうち、正しく予測された割合
# f1-score:適合率と再現率の調和平均ー＞今回のデータセットでは全ての値が1.0となり、非常に高い数値を示している(というか最大(0-1なので))
# support:各クラスの実際のデータポイント数