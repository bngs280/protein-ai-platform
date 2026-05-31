import argparse
from Bio import SeqIO
import json
import os

parser = argparse.ArgumentParser()

parser.add_argument("--fasta", required=True)
parser.add_argument("--mutation", required=True)

args = parser.parse_args()

record = SeqIO.read(args.fasta, "fasta")

sequence = str(record.seq)
print("New start")
print("Protein Loaded")
print("Name:", record.id)
print("Length:", len(sequence))

mutation = args.mutation

print("Mutation:", mutation)

# Dummy stability score
stability_score = -1.25

result = {
    "protein": record.id,
    "length": len(sequence),
    "mutation": mutation,
    "stability_score": stability_score
}

os.makedirs("output", exist_ok=True)

with open("output/result.json", "w") as f:
    json.dump(result, f, indent=4)

print("Analysis Complete")
print("Results saved to output/result.json")
