# [INTERSPEECH 2026] Domain-Aware Mispronunciation Detection and Diagnosis Using Language-Specific Statistical Graphs

[![arXiv](https://img.shields.io/badge/arXiv-2606.05569-b31b1b.svg)](https://arxiv.org/abs/2606.05569)

## 1. Setup Environment

From the repository root, run:

```bash
conda create -n mdd python==3.10.12
conda activate mdd
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 2. Generate Statistics

The statistics script is inside the `data/` folder. Run it from there so it can read `train.csv` and write the JSON outputs:

```bash
cd data
python stat.py
```

This will generate language-specific statistic files such as:

- `data_arabic.json`
- `data_mandarin.json`
- `data_hindi.json`
- `data_korean.json`
- `data_spanish.json`
- `data_vietnamese.json`

These files contain the computed confusion statistics for each language.

## 3. Prepare Audio Data

The code assumes all audio files referenced by `train.csv`, `dev.csv`, and `test.csv` exist under `EN_MDD/WAV/`.

- Copy or move all `.wav` files into `EN_MDD/WAV/`
- Make sure the `Path` column values in `data/train.csv`, `data/dev.csv`, and `data/test.csv` match the file names under `EN_MDD/WAV/`
- If `Path` includes subdirectories, preserve the same relative structure under `EN_MDD/WAV/`

## 4. Verify Audio Path Settings

The project currently uses a hardcoded audio base path in these files:

- `dataloader.py` line 32: `./EN_MDD/WAV/`
- `train.py` line 168: `./EN_MDD/WAV/`

If your audio files are stored elsewhere, update those paths accordingly.

## 5. Train the Model

From the repository root, run:

```bash
python train.py
```

Optional arguments:

```bash
python train.py --num_epoch 100 --batch_size 4 --lr 2e-5
```