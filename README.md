# generate_texture_image

テクスチャ画像をGANによって生成するプログラム

## プロジェクト構成

```bash
Project
  ├─run.ipynb              #実行用ノートブック
  ├─.gitignore
  ├─Models                 #作成モデルを保存
  ├─src                    #ソースファイルを格納しているフォルダ
     ├─PreProcess.py       #前処理用のファイル
     ├─Learn.py            #学習用のファイル
     ├─InferenceTest.py    #検証用のファイル
     ├─Models.py           #モデルの設定を行っているファイル
     ├─DataSets.py         #データローダーのためのファイル
  
```

## 特徴
- GANを用いた深層学習モデル
- PyTorchをベースにしたモデル

## 環境構築
### 1 リポジトリのクローン
```bash
   git clone git@github.com:Dakkahi/generate_texture_image.git
```

### 2 必要なパッケージのインストール
```bash
   pip install -r requirements.txt
```

### 3 データセットの所在
https://repository.upenn.edu/entities/publication/c1b8168b-b8a7-4f22-9e38-6430c233a600 のテクスチャデータセットを利用


