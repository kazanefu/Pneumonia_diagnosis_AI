```bash
conda create -n xray-ai python=3.10
conda activate xray-ai
```
ちゃんとactivateできてるか確認すること
powershellでは何もしないとactivateができなかったのでcmdを利用
powershellでやりたい場合は`conda init powershell`とすればいいらしい

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
とりあえずCPUで動かす
各種ライブラリのインストール
```bash
pip install pillow matplotlib
pip install scikit-learn tqdm
pip install onnx onnxruntime
```
| ライブラリ       | 役割          |
| ----------- | ----------- |
| torch       | NN本体        |
| torchvision | 画像処理・ResNet |
| Pillow     | 画像読み込み  |
| matplotlib | 学習曲線・確認 |
| sklearn | ROC-AUC / confusion matrix |
| tqdm    | 進捗バー                       |
| onnx        | モデル保存形式    |
| onnxruntime | Python側検証用 |

python/train.pyを書く
pythonに移動して
```bash
python train.py
```
これでpython/model/model.onnxができる
あとはRustで推論CLIアプリを作るだけ