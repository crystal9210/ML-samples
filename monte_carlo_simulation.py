import numpy as np
import matplotlib.pyplot as plt
# モンテカルロ法のサンプルコード

# モンテカルロ法：確率・統計を利用したランダムサンプリングを行う(外部ライブラリから乱数生成の関数を用いて
# 制約条件を付与したデータポイントを任意数生成する)ことで、数値の近似やシミュレーションを行う手法
# ー＞複雑な数値の近似や確率分布の推定が可能となる

# --- POINT ----
# モンテカルロ法としては直接計算をすることが難しい課題となる数値や条件に対応する関係式の妥当性を
# 一様分布から生成される乱数によるデータポイントの分布比から数式や求めたい条件に応じて計算をすることで
# 近似的に求めるー＞今回は例としてπを近似的に求めたー＞データポイント数num_samplesの整数値が大きくなればなるほど正確性が高まるはず、と予測される

# モンテカルロ法による円周率の推定
def monte_carlo_pi_simulation(num_samples):
    inside_circle = 0 # 円の中に入った点の数をカウントする変数初期化
    x_values = []
    y_values = []
    inside_x = []
    inside_y = []
    outside_x = []
    outside_y = []

    for _ in range(num_samples):
        x = np.random.rand() # 0から1の間のランダムな実数を生成ー＞生成される乱数は一様分布に従っている
        y = np.random.rand()
        x_values.append(x)
        y_values.append(y)
        if x**2 + y**2 <= 1:
            inside_circle += 1
            inside_x.append(x)
            inside_y.append(y)
        else:
            outside_x.append(x)
            outside_y.append(y)

    pi_estimate = (inside_circle / num_samples) * 4
    return pi_estimate, inside_x, inside_y, outside_x, outside_y


# シミュレーションの実行
num_samples = 100000
pi_estimate, x_inside, y_inside, x_outside, y_outside = monte_carlo_pi_simulation(num_samples)

print(f"Estimated Pi: {pi_estimate}") # π=3.141592...
# output:Estimated Pi: 3.1372 when num_samples = 10000
# output:Estimated Pi: 3.14288 when num_samples = 100000

plt.figure(figsize=(6, 6))
plt.scatter(x_inside, y_inside, color='blue', s=1, label='Inside Circle')
plt.scatter(x_outside, y_outside, color='red', s=1, label='Outside Circle')
plt.legend()
plt.title('Monte Carlo Simulation for Pi')
plt.show()
