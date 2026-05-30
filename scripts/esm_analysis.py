import torch
import esm
from Bio import SeqIO
import argparse
import json
import os

parser = argparse.ArgumentParser()

parser.add_argument("--fasta", required=True)
parser.add_argument("--mutation", required=True)

args = parser.parse_args()

# =========================
# LOAD FASTA
# =========================

record = SeqIO.read(args.fasta, "fasta")

sequence = str(record.seq)

print("Protein:", record.id)
print("Length:", len(sequence))

# =========================
# LOAD ESM MODEL
# =========================

print("Loading ESM2 model...")

model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()

batch_converter = alphabet.get_batch_converter()

model.eval()

# =========================
# PREPARE INPUT
# =========================

data = [
    ("protein1", sequence)
]

batch_labels, batch_strs, batch_tokens = batch_converter(data)

# =========================
# RUN MODEL
# =========================

print("Running AI inference...")

with torch.no_grad():
    results = model(batch_tokens, repr_layers=[6])

token_representations = results["representations"][6]

# Mean embedding
embedding = token_representations[0, 1:len(sequence)+1].mean(0)

embedding_vector = embedding.numpy().tolist()

print("Embedding generated")
print("Embedding size:", len(embedding_vector))

# =========================
# SIMPLE MUTATION SCORING
# =========================

mutation = args.mutation

# dummy AI-like score for now
mutation_score = float(torch.rand(1))

print("Mutation:", mutation)
print("Mutation impact score:", mutation_score)

# =========================
# SAVE OUTPUT
# =========================

result = {
    "protein": record.id,
    "length": len(sequence),
    "mutation": mutation,
    "embedding_size": len(embedding_vector),
    "mutation_score": mutation_score
}

os.makedirs("output", exist_ok=True)

outfile = f"output/{record.id}_analysis.json"

with open(outfile, "w") as f:
    json.dump(result, f, indent=4)

print("Saved:", outfile)
