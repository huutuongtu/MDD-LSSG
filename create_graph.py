import json
import networkx as nx
import matplotlib.pyplot as plt
import torch 
import numpy as np

l1 = ["arabic", "mandarin", "hindi", "korean", "spanish", "vietnamese"]


def get_graph(
    languages,
    alpha: float = 1.0,
):
    edges = {}
    weights = {}

    for lang in languages:
        data = json.load(open(f"./data/data_{lang}.json", "r", encoding="utf8"))
        
        by_i = {}  # i -> list[(j, cnt)]
        for key, value in data.items():
            i, j = map(int, key.split("_"))
            cnt = float(value)

            if cnt <= 0:
                continue
            if i == j:
                continue  

            by_i.setdefault(i, []).append((j, cnt))

        lang_edges = []
        lang_weights = []

        for i, items in by_i.items():
            denom = sum(cnt for (_, cnt) in items)
            if denom <= 0:
                continue
            dist = [(j, cnt / denom) for (j, cnt) in items]

            if alpha is not None and alpha != 1.0:
                s = sum((w ** alpha) for (_, w) in dist)
                if s > 0:
                    dist = [(j, (w ** alpha) / s) for (j, w) in dist]

            for j, w in dist:
                lang_edges.append((j, i))
                lang_weights.append(float(w))

        edges[lang] = lang_edges
        weights[lang] = lang_weights

    return edges, weights
