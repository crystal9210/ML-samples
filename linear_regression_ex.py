import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# データ生成
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# データを訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの学習
model = LinearRegression()
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# 精度の測定
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 平均二乗誤差（Mean Squared Error, MSE）->各データプロットに対する予測値と実測値の差の二乗の平均値を示す
# つまり、予測値が実測値から平均どの程度離れているかを単純に評価する
print(f"Mean Squared Error: {mse}")
# R2 score->決定係数；モデルの予測値が実行値とどの程度一致しているかを測る指標
# ー＞0から1まで数値をとり、1に近いほどモデルがデータをよく説明できていることを示す
print(f"R2 Score: {r2}")
# 結果のプロット
plt.scatter(X_test, y_test, color='black', label='Actual')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()
