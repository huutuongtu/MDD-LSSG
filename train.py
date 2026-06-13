import os
import gc
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import argparse

from jiwer import wer
import torch.nn.functional as F
from dataloader import (
    MDD_Dataset,
    collate_fn,
    PAD_ID,
    BLANK_ID,
    VOCAB_SIZE,
    text_to_tensor,
    L1GroupedBatchSampler
)
from gcn_model import GCN_MDD
from create_graph import get_graph
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor
import librosa
from pyctcdecode import build_ctcdecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--topk", type=int, default=None)
parser.add_argument("--min_prob", type=float, default=0)

parser.add_argument("--num_epoch", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lr", type=float, default=2e-5)

args = parser.parse_args()


def fmt(v):
    if v is None:
        return "none"
    if isinstance(v, float):
        return str(v).replace(".", "")
    return str(v)

ckpt_name = f"alpha{fmt(args.alpha)}"
CHECKPOINT_DIR = f"checkpoint_{ckpt_name}"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

gc.collect()

df_train = pd.read_csv("./data/train.csv")
df_dev = pd.read_csv("./data/dev.csv")

L1_LIST = ["arabic", "mandarin", "hindi", "korean", "spanish", "vietnamese"]

train_dataset = MDD_Dataset(df_train)

train_sampler = L1GroupedBatchSampler(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    seed=42
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    collate_fn=collate_fn,
)


all_edges, all_weights = get_graph(
    L1_LIST,
    alpha=args.alpha,
)


config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = GCN_MDD(
    config,
    vocab_size=VOCAB_SIZE,
    pad_id=PAD_ID,
).to(device)

model.wav2vec2.freeze_feature_extractor()


feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, padding_side='right',do_normalize=True, return_attention_mask=False)
dict_vocab = {"t ": 0, "uw ": 1, "er ": 2, "ah ": 3, "sh ": 4, "ng ": 5, "ow ": 6, "aw ": 7, "aa ": 8, "th ": 9, "ih ": 10, "zh ": 11, "k ": 12, "y ": 13, "l ": 14, "uh ": 15, "ch ": 16, "w ": 17, "b ": 18, "v ": 19, "ao ": 20, "s ": 21, "p ": 22, "iy ": 23, "r ": 24, "eh ": 25, "f ": 26, "n ": 27, "ay ": 28, "oy ": 29, "d ": 30, "g ": 31, "ey ": 32, "err ": 33, "dh ": 34, "ae ": 35, "hh ": 36, "m ": 37, "jh ": 38, "z ": 39, "/": 40, "": 41 }
decoder_ctc = build_ctcdecoder(
            labels = list(dict_vocab.keys()),
            )


# blank id is not important since we use transcript flat and transcript lengths for CTC loss, but we set it to the last index just in case 
ctc_loss = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scaler = torch.cuda.amp.GradScaler()
min_wer = 100

for epoch in range(args.num_epoch):
    train_sampler.set_epoch(epoch)
    model.train()
    epoch_losses = []

    print(f"EPOCH {epoch}")
    print(f"==============================")

    for step, batch in enumerate(tqdm(train_loader, leave=False, desc="Train")):
        try:
            audio, canonical, transcript_flat, transcript_lengths, batch_l1 = batch

            edge_index = torch.tensor(all_edges[batch_l1], dtype=torch.long).t().contiguous().to(device)
            edge_weight = torch.tensor(all_weights[batch_l1], dtype=torch.float).to(device)
            model.set_graph(edge_index, edge_weight)

            with torch.no_grad():
                audio_lengths = torch.full(
                    (audio.size(0),),
                    audio.size(1),
                    dtype=torch.long,
                    device=device
                )

            with torch.cuda.amp.autocast():
                logits = model(audio, canonical)
                log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
                input_lengths = torch.full(size=(log_probs.shape[1],), fill_value=log_probs.shape[0], dtype=torch.long, device=device)
                loss = ctc_loss(
                    log_probs,
                    transcript_flat,
                    input_lengths,
                    transcript_lengths
                )
            # print(loss)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())

        except Exception as e:
            print(f"[ERROR] step={step}: {e}")
            torch.cuda.empty_cache()
            continue
        
        # break

    if len(epoch_losses) == 0:
        print("No valid batches this epoch.")
        continue

    avg_loss = sum(epoch_losses) / len(epoch_losses)
    print(f"\nEpoch {epoch} mean CTC loss: {avg_loss:.4f}")

    # eval to keep best checkpoint
    # check avg_loss<=1 to decode faster
    if avg_loss<=1:
        with torch.no_grad():
            model.eval().to(device)
            worderrorrate = []
            for point in tqdm(range(len(df_dev))):
                acoustic, _ = librosa.load("./EN_MDD/WAV/" + df_dev['Path'][point] + ".wav", sr=16000)
                acoustic = feature_extractor(acoustic, sampling_rate = 16000)
                acoustic = torch.tensor(acoustic.input_values, device=device)
                transcript = df_dev['Transcript'][point]
                canonical = df_dev['Canonical'][point]
                l1 = df_dev['L1'][point].lower()

                edge_index = torch.tensor(all_edges[l1], dtype=torch.long).t().contiguous().to(device)
                edge_weight = torch.tensor(all_weights[l1], dtype=torch.float).to(device)
                model.set_graph(edge_index, edge_weight)

                canonical = text_to_tensor(canonical)
                canonical = torch.tensor(canonical, dtype=torch.long, device=device)
                logits = model(acoustic, canonical.unsqueeze(0))
                logits = F.log_softmax(logits.squeeze(0), dim=1)
                x = logits.detach().cpu().numpy()
                hypothesis = decoder_ctc.decode(x).replace("/", "").strip()
                error = wer(transcript, hypothesis)
                worderrorrate.append(error)
                # print(hypothesis)
                # break
            epoch_wer = sum(worderrorrate)/len(worderrorrate)

            if (epoch_wer < min_wer):
                print("save_checkpoint...")
                min_wer = epoch_wer
                ckpt_path = os.path.join(
                    CHECKPOINT_DIR,
                    f"best_{ckpt_name}.pth"
                )
                torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "alpha": args.alpha,
                },
                    ckpt_path
                )

            print("wer checkpoint " + str(epoch) + ": " + str(epoch_wer))
            print("min_wer: " + str(min_wer))