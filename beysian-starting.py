import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import random

# ベイズ推論の基礎を学習するためのサンプルコード
# サブプロットされる各図は異なる条件下での赤玉の割合(θ)の事後分布を示す
# ベータ分布のパラメータ(α,β)と試行回数(n)の組み合わせえ赤玉の割合がどのように変化するかを視覚的に表現


# --- POINT ---
# 試行回数が増加すると事後分布は非常に狭くなり、観測データに強く一致するようになる
# ベイズ推論の基本設定
balls = [1, 0, 0]  # 赤玉1、白玉0
n_trials = [0, 3, 10, 100]  # 試行回数のリスト(0回:観測データなし、3回:少数の試行回数、10回:中程度、1000回:多数の試行回数)

# --- POINT ---
# ベータ分布の分布関数；f(x;a,b)=(x^(a-1)*(1-x)^(b-1))/B(a,b) |
# a:成功回数に対する事前の信念ー＞値が大きいほど分布が右にシフトし成功確率が高いと信じていることを示す
# b:失敗回数に対する事前の信念ー＞値が大きいほど分布がhダリにシフトし、成功確率が低いと信じていることを示す
# 下記のbeta_paramsに渡すセット(a,b)の値の組み合わせに対する影響についてコメント
# a=1,b=1:一様分布。全ての確率が等しく信じられる（無知な事前分布）。例: どの確率も同じように可能性があると信じている場合。
# a>1,b=1:右にシフトした分布。成功確率が高い方が信じられている。例: 成功（赤玉）が多く観測されると信じている場合。
# a=1,b>1:左にシフトした分布。成功確率が低い方が信じられている。
# a>1,b>1:中央が高い分布。中程度の確率が信じられている。例: 確率が0.5付近であると信じている場合。
# a<1,b<1: 両端が高い分布。確率が極端に高いか低いかが信じられている。例: 確率が0か1に近いと信じている場合。
beta_params = [(0.5, 0.5), (1.0, 1.0), (5, 2)]  # ベータ分布のパラメータのリストー＞上の行から順に表示
idatum = []  # 結果を保存するリスト

# ベイズ推論のメイン処理
# 異なるベータ分布パラメータ試行回数の組み合わせに対して各々赤玉の割合の事後分布をサンプリングし結果をidatumリストに保存
for beta_param in beta_params:
    for n_trial in n_trials:
        with pm.Model() as red_white_model:
            # 赤玉を選ぶ事前確率としてベータ分布を仮定
            prior = pm.Beta("prior", alpha=beta_param[0], beta=beta_param[1])

            # 赤玉が出た回数を乱数で計算して観測値とする
            n_red = sum(random.choice(balls) for _ in range(n_trial))

            # 尤度関数は二項分布 | 各引数ー＞1:確率変数名、2:二項分布の成功確率p、3:試行回数、4:観測データ
            # 尤度:成功確率pに基づいてn回成功する確率を計算する
            # obs:PyMC3モデル内の確率変数、二項分布に従う観測データを表現i.e.観測データに基づいて定義される二項分布
            # obsは直接参照されることはないが、観測データに基づく尤度を定義するために使用し、間接的にモデル全体の一部として機能している.sample()の部分
            obs = pm.Binomial("obs", p=prior, n=n_trial, observed=n_red)
            # 赤玉の割合θの事後分布をサンプリング
            idata = pm.sample() # 事後分布からサンプルを生成するための関数
            idatum.append(idata)

            # 事後分布のサンプルを表示
            print("Posterior samples:")
            print(az.summary(idata, var_names=["prior"]))


# 結果のプロット
fig, axs = plt.subplots(3, 4, figsize=(15, 8))
plt.subplots_adjust(hspace=1)

count = 0
for i in range(12):
    ii = i // 4
    jj = i % 4
    ax = axs[ii, jj]
    ax.set_xlim(xmin=0, xmax=1)
    ax.set_frame_on(True)
    az.plot_posterior(
        idatum[count],
        group="posterior",
        hdi_prob="hide",
        point_estimate=None,
        ax=ax
    )
    ax.text(0.8, 1.0, f"N={n_trials[jj]}")
    count += 1

plt.show()
