import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed



gap_penalty = -1
match_award = 1
mismatch_penalty = -1
def zeros(rows, cols):
    # Define an empty list
    retval = []
    # Set up the rows of the matrix
    for x in range(rows):
        # For each row, add an empty list
        retval.append([])
        # Set up the columns in each row
        for y in range(cols):
            # Add a zero to each column in each row
            retval[-1].append(0)
    # Return the matrix of zeros
    return retval

def match_score(alpha, beta):
    if alpha == beta:
        return match_award
    elif alpha == "<eps>" or beta == "<eps>":
        return gap_penalty
    else:
        return mismatch_penalty

def Align(seq1, seq2):
    
    # Store length of two sequences
    n = len(seq1)  
    m = len(seq2)
    
    # Generate matrix of zeros to store scores
    score = zeros(m+1, n+1)
   
    # Calculate score table
    
    # Fill out first column
    for i in range(0, m + 1):
        score[i][0] = gap_penalty * i
    
    # Fill out first row
    for j in range(0, n + 1):
        score[0][j] = gap_penalty * j
    
    # Fill out all other values in the score matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Calculate the score by checking the top, left, and diagonal cells
            match = score[i - 1][j - 1] + match_score(seq1[j-1], seq2[i-1])
            delete = score[i - 1][j] + gap_penalty
            insert = score[i][j - 1] + gap_penalty
            # Record the maximum score from the three possible scores calculated above
            score[i][j] = max(match, delete, insert)
    
    # Traceback and compute the alignment 
    
    # Create variables to store alignment
    align1 = []
    align2 = []
    
    # Start from the bottom right cell in matrix
    i = m
    j = n
    
    # We'll use i and j to keep track of where we are in the matrix, just like above
    while i > 0 and j > 0: # end touching the top or the left edge
        score_current = score[i][j]
        score_diagonal = score[i-1][j-1]
        score_up = score[i][j-1]
        score_left = score[i-1][j]
        
        # Check to figure out which cell the current score was calculated from,
        # then update i and j to correspond to that cell.
        if score_current == score_diagonal + match_score(seq1[j-1], seq2[i-1]):
            align1.append(seq1[j-1])
            align2.append(seq2[i-1])
            i -= 1
            j -= 1
        elif score_current == score_up + gap_penalty:
            align1.append(seq1[j-1])
            align2.append("<eps>")
            j -= 1
        elif score_current == score_left + gap_penalty:
            align1.append("<eps>")
            align2.append(seq2[i-1])
            i -= 1

    # Finish tracing up to the top left cell
    while j > 0:
        align1.append(seq1[j-1])
        align2.append("<eps>")
        j -= 1
    while i > 0:
        align1.append("<eps>")
        align2.append(seq2[i-1])
        i -= 1
    
    # Since we traversed the score matrix from the bottom right, our two sequences will be reversed.
    # These two lines reverse the order of the characters in each sequence.
    align1 = align1[::-1]
    align2 = align2[::-1]
    
    return(align1, align2)


def confusion_matrix(seq1: str, seq2: str, token=None) -> dict:
    aligned_seq1, aligned_seq2 = Align(seq1.split(" "), seq2.split(" "))
    pairs = []
    for i in range(len(aligned_seq1)):
        if aligned_seq1[i] == token:
            # Remove self-loop: token -> token
            #if aligned_seq1[i] != aligned_seq2[i]:
            pairs.append((aligned_seq1[i], aligned_seq2[i]))

    unique_pairs = set(pairs)
    counts = {}
    for pair in unique_pairs:
        counts[pair] = pairs.count(pair)
    return counts



def merge_2dict(res: dict, dict_need_to_merge: dict) -> dict:
    result = {}
    for key in res:
        if key in dict_need_to_merge:
            result[key] = res[key] + dict_need_to_merge[key]
        else:
            result[key] = res[key]

    for key in dict_need_to_merge:
        if key not in res:
            result[key] = dict_need_to_merge[key]
    
    return result

def get_keys_by_value(my_dict, target_value):
    return [key for key, value in my_dict.items() if value == target_value]

# dataset EDA
data = pd.read_csv("train.csv")

vocab = {"t": 0, "uw": 1, "er": 2, "ah": 3, "sh": 4, "ng": 5, "ow": 6, "aw": 7, "aa": 8, "th": 9, "ih": 10, "zh": 11, "k": 12, "y": 13, "l": 14, "uh": 15, "ch": 16, "w": 17, "b": 18, "v": 19, "ao": 20, "s": 21, "p": 22, "iy": 23, "r": 24, "eh": 25, "f": 26, "n": 27, "ay": 28, "oy": 29, "d": 30, "g": 31, "ey": 32, "err": 33, "dh": 34, "ae": 35, "hh": 36, "m": 37, "jh": 38, "z": 39, "<eps>": 40}

L1_LIST = [
    "Arabic",
    "Mandarin",
    "Hindi",
    "Korean",
    "Spanish",
    "Vietnamese",
]


for l1 in L1_LIST:
    print(f"Processing L1: {l1}")
    
    subset = data[data["L1"] == l1]
    res = {}

    for phoneme in tqdm(vocab.keys(), desc=l1):
        for i in range(len(subset)):
            res = merge_2dict(
                res,
                confusion_matrix(
                    subset.iloc[i]["Canonical"],
                    subset.iloc[i]["Transcript"],
                    token=phoneme
                )
            )

    # Convert phoneme → id
    numeric_data = {
        (vocab[t1], vocab[t2]): count
        for (t1, t2), count in res.items()
    }

    # Make JSON compatible
    json_data = {
        f"{k[0]}_{k[1]}": v
        for k, v in numeric_data.items()
    }

    with open(f"data_{l1.lower()}.json", "w") as f:
        json.dump(json_data, f, indent=4)



