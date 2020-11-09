Classification and Regression: In a Weekend 

memo by ksnt

# EDA

## 記述統計

describe()関数を使うと記述統計量が見れる。
.shape属性でデータの形を見ることができる。

### 数値的記述統計量

平均とか標準偏差とか。標準偏差は平均からの散らばりを計算。


## グラフィカルな記述統計量

### ヒストグラムとボックスプロット - 分布を理解する

ヒストグラムは回数や頻度、'bin'のあいだで生じる値を教えてくれる

ボックスプロットは「25パーセンタイルから75パーセンタイルの範囲とメディアン」を教えてくれる。まら、各々の特徴量の「最小値」と「最大値」を捉えている。また、これらのチャートは各々の特徴量の値の分布を教えてくれる。これを見て、どのようにデータを扱うのかの判断をはじめることができる。例えば外れ値を処理したいかどうかということやデータを正規化したいかどうかということ。

### ボックスプロットとIQR

平均と標準偏差の代わりにメディアンとinterquartile range(IQR)を使う。IQRは75th quantileと25th quantileの差である。

IQRはデータの真ん中50%がどこか教えてくるが、一方で標準偏差はデータのちらばりについて教えてくれる。

メディアンとIQRは外れ値にロバストである。

メディアンと四分位点の値を比べるとデータが歪んでいるかどうかが分かる（らしい）。

## 相関

最も広く使われている方法

1.Pearson Correlation Coefficient 
2.Spearman's Correlation
3.Kendall's Tau


1. Pearson

ピアソン相関は連続変数間の線形な連関を測定する。言い換えると、二つの変数の間の関係がどの程度直線によって記述できるかということをはかる。


# Pre-processing data

## 欠損値を扱う

pandasには欠損値を埋めるためのさまざまなオプションが用意されている。また、欠損値が何であるべきかを予測するためにk近傍法やsklearn imputer関数を使うことが出来る。

## カテゴリカル値の処置

カテゴリカル値を数値に変換する。sklearnやpandasにはたくさんの方法がある。


### データの正規化

Normalization: 変数のが0から1の間に入るようにする
Standardization: データを平均0、標準偏差1にする

機械学習のプリプロセッシングタスクにおいてはこのような方法でデータをリスケーリングすることはよくある：なぜなら、多くのアルゴリズムは全ての特徴量が同じスケールであるということを仮定しているから。大抵は0から1あるいは-1から1。

scikitlearnではMinMaxScalerとStandardScalerがよく使われる。

- MinMaxScaler

z = (x - min(x)) / (max(x) - min(x))

- StandardScaler

(x_i - mean(x)) / stdev(x)


### データの分割

オリジナルのデータセットは訓練データとテストデータに分けられるべきである。

訓練データ: このデータはモデルを作るために用いられる - 例)線形回帰において最適な相関係数を見つける, 決定木をつくるためにCARTアルゴリズムを使う

テストデータ: このデータは未見のデータに対してモデルがどれくらいのパフォーマンスを出すのかを見るために用いる

scikitlearnに色んな道具が用意されている（ぽい）


#ベースラインアルゴリズムを選ぶ

## ベースラインモデルを定義する

ベースラインというのはデータセットのための予測をつくるためにヒューリスティクス、簡単なサマリ統計、ランダムネス、あるいは機械学習を使う方法

## 開発したモデルをトレーニングセットにフィッティングする

モデルフィッティングは3つのステップが含まれる手続きである
1. まずはじめにパラメタのセットを取り予測されたデータセットを返す関数が必要
2. 次に、あなたのデータとモデルの予測の間の差を表す数を与える「誤差関数」が必要である。これは大抵 sums of squared error(SSE)かmaximum likelihood(最大尤度)である。
3. 最後にこの差を最小化するパラメタを見つける必要がある


## 評価基準を定義する

回帰タスクに最もよく使われる基準はRMSE(root-mean-square-error)である。

RMSE = \sqrt(\sigma_i (y_i - y^hatto_i)^2 )

Mean Squared Error: 推測された値と結果として得たものの差

Mean Absolute Error: 二つの連続変数の間の差を測定したもの

R^2: どれくらいよく実際の結果がモデルあるいは回帰直線によって再現されるか


# 分類のための評価基準

分類のためのよくある評価基準はAccuracyである。

Accuracy = # correct predictions / # total data points

Accuracyはunbalanced classには向いていない。そこで、confusion matrixを考える。

Accuracy = (TruePositives + FalseNegatives) / TotalNumberofSamples

- Area Under Curve (AUC)

ROC曲線と直線の間の面積(AUC)で分類モデルの良し悪しを決める

数値の場合はF1スコアを使う

# モデルを改善する - ベースラインモデルから最終モデルへ

ベースラインモデルを改善するための戦略は以下の通り:
a) Feature engineering –   by adding extra columns and trying to understand if it improves the model
b) Regularization to prevent overfitting
c) Ensembles –   typically for classificatio
d) Test alternative models
e) Hyperparameter tuning


## クロスバリデーションを理解する

通常5-foldか10-foldが使われるがルールはない
sklearnのKFold()を使う

### Classification code outline
Load the data
Exploratory data analysis
    Analyse the target variable 
    Check if the data is balanced 
    Check the co-relations
Split the data
Choose a Baseline algorithm
Train and Test the Model
Choose an evaluation metric
Refine our dataset 
Feature engineering
Test Alternative Models
Ensemble models
Choose the best model and optimise its parameters

1) Choosing alternate models:

もし二つのモデルがあり、どちらがより良いモデルなのかということが知りたいのならば与えられたデータセットに対してその二つを比較するためにクロスバリデーションを用いることが出来る。

"""### Test Alternative Models
logistic = LogisticRegression()
cross_val_score(logistic, X, y, cv=5, scoring="accuracy").mean()
rnd_clf = RandomForestClassifier()
cross_val_score(rnd_clf, X, y, cv=5, scoring="accuracy").mean()

2) hyperparameter tuning 

最終的に、クロスバリデーションはハイパーパラメーターチューニングにおいてもまた使われる
As per cross validation parameter tuning grid search“In  machine  learning,  two  tasks  are  commonly  done  at  the  same   time   in   data   pipelines:   cross   validation   and   (hy-per)parameter  tuning.  Cross  validation  is  the  process  of  train-ing learners using one set of data and testing it using a different set. Parameter tuning is the process to selecting the values for a model’s parameters that maximize the accuracy of the model.”
結論付けると、クロスバリデーションはデータサイエンスパイプラインの複数の箇所で使われるテクニックである。

## Feature engineering

特徴量エンジニアリングは機械学習パイプラインの重要なパートである。
我々は我々のデータセットを追加カラムを付け加えることで改良する。その目的は、特徴量のあるコンビネーションは問題の空間をよりよく表したり目的変数のindicatorになるからである。
pandasを使ってカラムを追加するには以下のようにやればよい:
boston_X['LSTAT_2'] = boston_X['LSTAT'].map(lambda x: x**2)
何か改善したかどうかを確かめるために改良される前のデータセットに対してと同様のコードを走らせることが出来る:
lm.fit(X_train, Y_train) 
Y_pred = lm.predict(X_test) 
evaluate(Y_test, Y_pred)


## Regularization to prevent overfitting

min_f \sigma V(f(x_i), y_i) + \lambda R(f)

As the value of λ rises, it reduces the value of coefficients and thus reducing the variance. 

ディープラーニングでは多くのregularizationテクニックが使われている: Dataset augmentation, Early stopping, Dropout layer, Weight penalty L1 and L2


## Ensembles –   typically for classification

Ensemble  learning  is  a  machine  learning  paradigm  where  mul-tiple models (often called “weak learners”) are trained to solve the same  problem  and  combined  to  get  better  results.

弱学習器を連結するメタアルゴリズムとして3つの主要な方法がある:

- bagging, that  often  considers  homogeneous  weak  learners,  learns  them  independently  from  each  other  in  parallel  and  combines  them following some kind of deterministic averaging process

- boosting, that often considers homogeneous weak learners, learns them sequentially in an adaptive way (a base model depends on the previous ones) and combines them following a deterministic strat-egy

- stacking, that often considers heterogeneous weak learners, learns them  in  parallel  and  combines  them  by  training  a  meta-model  to  output a prediction based on the different weak models predictions

弱学習器はよりよいパフォーマンスを持つモデルを得るために連結することができる。


## Test alternative models

"""### Test Alternative Models logistic = LogisticRegression() cross_val_score(logistic, X, y, cv=5, scoring="accuracy").mean() rnd_clf = RandomForestClassifier() cross_val_score(rnd_clf, X, y, cv=5, scoring="accuracy").mean()

## Hyperparameter tuning

ハイパーパラメータチューニングの目標は、ベストパフォーマンスになる設定を見つけるためにさまざまなハイパーパラメータを横断して探ることである

ハイパーパラメータチューニングの主な方法としては以下の3つが挙げられる: Grid search, Random search and Bayesian optimisation


