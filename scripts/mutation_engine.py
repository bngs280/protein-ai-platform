import torch
import esm
import argparse
import numpy as np

from Bio import SeqIO
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# ARGUMENTS
# =========================

parser = argparse.ArgumentParser()

parser.add_argument("--fasta", required=True)
parser.add_argument("--mutation", required=True)

args = parser.parse_args()

# =========================
# LOAD FASTA
# =========================

record = SeqIO.read(args.fasta, "fasta")

protein_name = record.id
wt_sequence = str(record.seq)

print("\nProtein:", protein_name)
print("WT Length:", len(wt_sequence))

# =========================
# PARSE MUTATION
# Example: R175H
# =========================

mutation = args.mutation

wt_aa = mutation[0]
mut_aa = mutation[-1]
position = int(mutation[1:-1])

print("\nMutation:", mutation)

# Validate position
if position < 1 or position > len(wt_sequence):
    raise ValueError("Mutation position outside sequence")

# Validate WT amino acid
real_wt = wt_sequence[position - 1]

if real_wt != wt_aa:
    raise ValueError(
        f"WT amino acid mismatch. Sequence has '{real_wt}' at position {position}"
    )

# =========================
# CREATE MUTANT SEQUENCE
# =========================

mut_sequence = (
    wt_sequence[:position - 1]
    + mut_aa
    + wt_sequence[position:]
)

print("WT AA:", wt_aa)
print("MUT AA:", mut_aa)

# =========================
# LOAD ESM MODEL
# =========================

print("\nLoading ESM2 model...")

model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()

batch_converter = alphabet.get_batch_converter()

model.eval()

# =========================
# FUNCTION TO GENERATE EMBEDDING
# =========================

def get_embedding(sequence, position):

    data = [("protein", sequence)]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():

        results = model(
            batch_tokens,
            repr_layers=[6]
        )

    token_representations = results["representations"][6]

    residue_embedding = token_representations[
        0,
        position
    ]

    return residue_embedding.numpy()

# =========================
# GENERATE EMBEDDINGS
# =========================

print("\nGenerating WT embedding...")

wt_embedding = get_embedding(wt_sequence, position)

print("Generating MUT embedding...")

mut_embedding = get_embedding(mut_sequence, position)

# =========================
# COSINE SIMILARITY
# =========================

similarity = cosine_similarity(
    [wt_embedding],
    [mut_embedding]
)[0][0]

# =========================
# IMPACT SCORE
# =========================

impact_score = 1 - similarity

# =========================
# RESULTS
# =========================

print("\n========== RESULTS ==========")

print("Cosine Similarity:", round(float(similarity), 4))

print("Mutation Impact Score:", round(float(impact_score), 4))

# =========================
# INTERPRETATION
# =========================

if impact_score < 0.05:
    interpretation = "Likely benign"

elif impact_score < 0.15:
    interpretation = "Possibly impactful"

else:
    interpretation = "Potentially damaging"

print("Interpretation:", interpretation)

print("============================\n")
