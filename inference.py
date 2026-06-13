import os
import gc
import librosa
import torch
import pandas as pd
from tqdm import tqdm
from transformers import Wav2Vec2Config
from dataloader import (
    text_to_tensor,
    dict_vocab,
    PAD_ID,
    VOCAB_SIZE,
    feature_extractor,
)
from gcn_model import GCN_MDD
from create_graph import get_graph  
from pyctcdecode import build_ctcdecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CHECKPOINT_DIR = "checkpoint"
gc.collect()


id_to_token = {v: k for k, v in dict_vocab.items()}
dict_vocab = {"t ": 0, "uw ": 1, "er ": 2, "ah ": 3, "sh ": 4, "ng ": 5, "ow ": 6, "aw ": 7, "aa ": 8, "th ": 9, "ih ": 10, "zh ": 11, "k ": 12, "y ": 13, "l ": 14, "uh ": 15, "ch ": 16, "w ": 17, "b ": 18, "v ": 19, "ao ": 20, "s ": 21, "p ": 22, "iy ": 23, "r ": 24, "eh ": 25, "f ": 26, "n ": 27, "ay ": 28, "oy ": 29, "d ": 30, "g ": 31, "ey ": 32, "err ": 33, "dh ": 34, "ae ": 35, "hh ": 36, "m ": 37, "jh ": 38, "z ": 39, "/": 40, "": 41 }

df_test = pd.read_csv("./data/test.csv")

decoder_ctc = build_ctcdecoder(
                              labels = list(dict_vocab.keys()),
                              )


L1_LIST = ["arabic", "mandarin", "hindi", "korean", "spanish", "vietnamese"]


def fmt(v):
    if v is None:
        return "none"
    if isinstance(v, float):
        return str(v).replace(".", "")
    return str(v)

alpha = 1.0
all_edges, all_weights = get_graph(
    L1_LIST,
    alpha=alpha,
)



ckpt_path = f"./checkpoint_alpha10/best_alpha10.pth"

config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = GCN_MDD(
    config,
    vocab_size=VOCAB_SIZE,
    pad_id=PAD_ID,
).to(device)

ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

all_results = []

with torch.no_grad():
    for i in tqdm(range(len(df_test)), desc="Inference"):
        wav_path = "EN_MDD/WAV/" + df_test.loc[i, "Path"] + ".wav"
        wav, _ = librosa.load(wav_path, sr=16000)

        # wav2vec2 input
        inputs = feature_extractor(wav, sampling_rate=16000)
        audio = torch.tensor(inputs.input_values, device=device).float()  # (1, T)
        audio_mask = torch.ones_like(audio, dtype=torch.long, device=device)

        # canonical prompt
        canonical_str = df_test.loc[i, "Canonical"]
        canonical_ids = text_to_tensor(canonical_str)
        canonical = torch.tensor(canonical_ids, dtype=torch.long, device=device).unsqueeze(0)
        l1 = df_test["L1"][i].lower()


        edge_index = torch.tensor(all_edges[l1], dtype=torch.long).t().contiguous().to(device)
        edge_weight = torch.tensor(all_weights[l1], dtype=torch.float).to(device)
        model.set_graph(edge_index, edge_weight)

        # forward
        logits = model(audio, canonical)
        log_probs = logits.log_softmax(dim=-1).squeeze(0)

        x = log_probs.detach().cpu().numpy()
        hypothesis = decoder_ctc.decode(x).strip()


        all_results.append({
            "Path" : df_test.loc[i, "Path"],
            "L1" : df_test.loc[i, "L1"],  
            "Canonical" : canonical_str,
            "Transcript" : df_test.loc[i, "Transcript"],
            "Predict" : hypothesis,
        })


out_df = pd.DataFrame(all_results)
out_path = f"./predictions_gl1_{fmt(alpha)}.csv"
out_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
