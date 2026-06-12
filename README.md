## Buoc 1: Tao Colab Runtime dung

1. Su dung T4 GPU
2. Dat file zip cua du an vao /content

## Buoc 2: Setup moi truong

Chay:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Sau do chay:

```bash
!python scripts/colab_setup.py --drive-dir DA_NPL_Artifacts
```

Neu Drive da mount san, script se tu dung lai mount hien co.

Script nay se:
- su dung Google Drive tai `/content/drive` de backup
- cai dependencies tu `requirements.txt`
- clone va build KenLM
- cai KenLM Python bindings
- ghi manifest moi truong vao `artifacts/colab/environment.json`

Sau khi setup xong, tat ca artifacts van duoc ghi tren local disk cua Colab de toc do nhanh nhat. Google Drive chi dung de backup.

## Buoc 3: Chuan bi du lieu va vocab

Chay:

```bash
!python scripts/colab_prepare_data.py \
  --vocab-size 30000 \
  --min-freq 2 \
  --vocab-limit-lines 150000 \
  --drive-dir DA_NPL_Artifacts
```

Ket qua:
- dataset Wikitext-103 se nam o `data/wikitext-103/`
- vocab LSTM nam o `artifacts/lstm/vocab.pkl`
- manifest du lieu nam o `artifacts/colab/dataset_manifest.json`
- cac tep can thiet se duoc copy len Drive

## Buoc 4: Train KenLM

KenLM la baseline thong ke thay cho N-gram

Chay:

```bash
!python scripts/train_ngram.py \
  --train-path data/wikitext-103/wiki.train.tokens \
  --max-n 3 \
  --memory 50% \
  --candidate-vocab-size 30000 \
  --out artifacts/ngram/kenlm_word.arpa.json
```

Artifacts tao ra:
- `artifacts/ngram/kenlm_word.arpa.json`
- `artifacts/ngram/kenlm_word.arpa`
- `artifacts/ngram/kenlm_word.bin`
- `artifacts/ngram/kenlm_word.vocab.txt`
- `artifacts/ngram/kenlm_word.normalized.txt`

## Buoc 5: Fine-tune GPT-2

Chay:

!pip install --upgrade transformers accelerate peft
!pip install --upgrade torch torchvision torchaudio

```bash
!python scripts/colab_train_gpt2.py \
  --max-lines 50000 \
  --max-steps 1000 \
  --drive-dir DA_NPL_Artifacts
```

Khuyen nghi:
- de `max-lines=50000` cho budget 4 gio
- `--max-steps 1500` - ? phut
- `--max-steps 1000` - 7 phut

Artifacts GPT-2:
- `artifacts/gpt2_finetuned/checkpoint-*`
- `artifacts/gpt2_finetuned/final_model/`
- `artifacts/gpt2_finetuned/training_results.json`

## Buoc 6: Train LSTM tren T4

Config an toan cho 4 gio tong budget:

```bash
!python scripts/colab_train.py \
  --embed-size 256 \
  --hidden-size 512 \
  --num-layers 2 \
  --dropout 0.4 \
  --batch-size 256 \
  --epochs 12 \
  --seq-length 35 \
  --learning-rate 5e-4 \
  --weight-decay 1e-5 \
  --grad-clip 5.0 \
  --early-stopping 3 \
  --train-limit-lines 150000 \
  --valid-limit-lines 20000 \
  --device cuda \
  --drive-dir DA_NPL_Artifacts
```

Luu y:
- chi bat `--tie-weights` khi `embed-size == hidden-size`
- `train-limit-lines` la tham so quan trong nhat de khong vuot budget
- voi T4, `batch-size 256` la muc can bang tot giua toc do va RAM

Artifacts LSTM:
- `artifacts/lstm/best_model.pth`
- `artifacts/lstm/training_history.json`
- `artifacts/colab/lstm_train_manifest.json`

## Buoc 7: Evaluate va benchmark

Sau khi co KenLM, GPT-2 va LSTM, chay:

```bash
!python scripts/colab_evaluate.py \
  --ngram-model artifacts/ngram/kenlm_word.arpa.json \
  --lstm-model artifacts/lstm/best_model.pth \
  --lstm-vocab artifacts/lstm/vocab.pkl \
  --gpt2-model-name artifacts/gpt2_finetuned/final_model \
  --limit-lines 500 \
  --n-samples 100 \
  --drive-dir DA_NPL_Artifacts
```

Script se tu dong:
- evaluate KenLM
- evaluate LSTM
- evaluate GPT-2 fine-tuned
- tao qualitative report
- chay benchmark tong hop
- dong bo artifacts len Google Drive

Artifacts quan trong sau cung:
- `artifacts/benchmark.json`
- `artifacts/qualitative_100.json`
- `artifacts/lstm/test_results.json` neu co
- `artifacts/colab/evaluation_manifest.json`
