import torch
import esm
import argparse
import numpy as np

from Bio import SeqIO
from scipy.spatial.distance import cosine

# =========================
# ARGUMENTS
# =========================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--fasta",
    required=True
)

parser.add_argument(
    "--mutation",
    required=True
)

args = parser.parse_args()

# =========================
# LOAD FASTA
# =========================

record = SeqIO.read(args.fasta, "fasta")

protein_name = record.id
wt_sequence = str(record.seq)

# =========================
# PARSE MUTATION
# Example: G12D
# =========================

mutation = args.mutation

wt_aa = mutation[0]
mut_aa = mutation[-1]
position = int(mutation[1:-1])

# =========================
# VALIDATE
# =========================

if wt_sequence[position - 1] != wt_aa:

    print("ERROR:")
    print("Mutation mismatch with sequence")

    print(
        "Expected:",
        wt_sequence[position - 1]
    )

    exit()

# =========================
# CREATE MUTANT
# =========================

mut_sequence = list(wt_sequence)

mut_sequence[position - 1] = mut_aa

mut_sequence = "".join(mut_sequence)

print("\nProtein:", protein_name)

print("Mutation:", mutation)

# =========================
# LOAD ESM2
# =========================

print("\nLoading ESM2 model...")

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

model.eval()

batch_converter = alphabet.get_batch_converter()

# =========================
# EMBEDDING FUNCTION
# =========================

def get_embedding(sequence):

    data = [("protein", sequence)]

    _, _, tokens = batch_converter(data)

    with torch.no_grad():

        results = model(
            tokens,
            repr_layers=[33]
        )

    embedding = results["representations"][33]

    embedding = embedding.mean(1)

    return embedding.squeeze().numpy()

# =========================
# GENERATE EMBEDDINGS
# =========================

print("Generating WT embedding...")

wt_emb = get_embedding(wt_sequence)

print("Generating MUT embedding...")

mut_emb = get_embedding(mut_sequence)

# =========================
# CALCULATE SCORES
# =========================

similarity = 1 - cosine(
    wt_emb,
    mut_emb
)

impact_score = 1 - similarity

# heuristic stability score
stability_score = impact_score * 100

# =========================
# INTERPRETATION
# =========================

if stability_score < 5:

    interpretation = "Stable / likely benign"

elif stability_score < 15:

    interpretation = "Moderate destabilization"

else:

    interpretation = "Strong destabilization"

# =========================
# RESULTS
# =========================

print("\n========== RESULTS ==========")

print(
    "Cosine Similarity:",
    round(similarity, 4)
)

print(
    "Mutation Impact Score:",
    round(impact_score, 4)
)

print(
    "Predicted Stability Change:",
    round(stability_score, 2),
    "%"
)

print(
    "Interpretation:",
    interpretation
)

print("============================\n")