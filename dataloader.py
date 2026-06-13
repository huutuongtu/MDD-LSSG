import torch
from torch.utils.data import Dataset, Sampler
from collections import defaultdict
import math
import random
import numpy as np
import librosa
from transformers import Wav2Vec2FeatureExtractor


dict_vocab = {"t": 0, "uw": 1, "er": 2, "ah": 3, "sh": 4, "ng": 5, "ow": 6, "aw": 7, "aa": 8, "th": 9, "ih": 10, "zh": 11, "k": 12, "y": 13, "l": 14, "uh": 15, "ch": 16, "w": 17, "b": 18, "v": 19, "ao": 20, "s": 21, "p": 22, "iy": 23, "r": 24, "eh": 25, "f": 26, "n": 27, "ay": 28, "oy": 29, "d": 30, "g": 31, "ey": 32, "err": 33, "dh": 34, "ae": 35, "hh": 36, "m": 37, "jh": 38, "z": 39, "<eps>": 40, "<blank>": 41 }

PAD_ID = dict_vocab["<eps>"]
BLANK_ID = dict_vocab["<blank>"]
VOCAB_SIZE = len(dict_vocab)


def text_to_tensor(string_text):
    text = string_text.split(" ")
    return [dict_vocab[t] for t in text]


class MDD_Dataset(Dataset):
    def __init__(self, data):
        self.len_data   = len(data)
        self.path       = list(data['Path'])
        self.canonical  = list(data['Canonical'])
        self.transcript = list(data['Transcript'])
        self.l1         = [str(x).lower() for x in list(data['L1'])]

    def __getitem__(self, index):
        waveform, _ = librosa.load("./EN_MDD/WAV/" + self.path[index] + ".wav", sr=16000)
        canonical_ids  = text_to_tensor(self.canonical[index])
        transcript_ids = text_to_tensor(self.transcript[index])
        l1 = self.l1[index]
        return waveform, canonical_ids, transcript_ids, l1


    def __len__(self):
        return self.len_data


feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    padding_side='right',
    do_normalize=True,
    return_attention_mask=False
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_fn(batch, pad_id=PAD_ID):
    wavs, canonicals, transcripts, l1s = zip(*batch)
    assert len(set(l1s)) == 1, f"Mixed-language batch: {set(l1s)}"
    batch_l1 = l1s[0]
    # pad waveforms
    max_wav = max(len(w) for w in wavs)
    padded_wavs = []
    wav_lengths = []

    for w in wavs:
        wav_lengths.append(len(w))
        padded_wavs.append(
            np.pad(w, (0, max_wav - len(w)))
        )

    inputs = feature_extractor(padded_wavs, sampling_rate=16000)
    input_values = torch.tensor(inputs.input_values).float().to(device)

    # canonical: used as prompt (pad to max length)
    max_can = max(len(c) for c in canonicals)
    canonical_pad = [
        c + [pad_id] * (max_can - len(c))
        for c in canonicals
    ]
    canonical = torch.tensor(canonical_pad).long().to(device)

    # transcript: CTC target (NO padding tensor)
    transcript_flat = []
    transcript_lengths = []

    for t in transcripts:
        transcript_flat.extend(t)
        transcript_lengths.append(len(t))

    transcript_flat = torch.tensor(transcript_flat).long().to(device)
    transcript_lengths = torch.tensor(transcript_lengths).long().to(device)

    return (
        input_values,
        canonical,
        transcript_flat,
        transcript_lengths,
        batch_l1
    )



class L1GroupedBatchSampler(Sampler[list]):
    def __init__(self, dataset: MDD_Dataset, batch_size: int, shuffle: bool = True, drop_last: bool = True, seed: int = 1234):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        self.groups = defaultdict(list)
        for idx in range(len(dataset)):
            self.groups[dataset.l1[idx]].append(idx)

        self.l1_list = sorted(list(self.groups.keys()))
    
    def set_epoch(self, epoch):
        self.seed = epoch

    def __iter__(self,):
        rng = random.Random(1234 + self.seed)
        group_indices = {}
        for l1, idxs in self.groups.items():
            idxs = idxs.copy()
            if self.shuffle:
                rng.shuffle(idxs)
            group_indices[l1] = idxs

        # build batches per language
        batches = []
        for l1 in self.l1_list:
            idxs = group_indices[l1]
            n = len(idxs)
            num_full = n // self.batch_size
            rem = n % self.batch_size

            for k in range(num_full):
                b = idxs[k*self.batch_size:(k+1)*self.batch_size]
                batches.append(b)
            if rem and not self.drop_last:
                batches.append(idxs[num_full*self.batch_size:])

        if self.shuffle:
            rng.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self):
        total = 0
        for l1 in self.l1_list:
            n = len(self.groups[l1])
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += math.ceil(n / self.batch_size)
        return total







