from transformers import EsmForProteinFolding, AutoTokenizer
from transformers.models.esm.openfold_utils.protein import to_pdb
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

import torch
from Bio import SeqIO

# =========================
# LOAD FASTA
# =========================

record = SeqIO.read("test_data/kras.fasta", "fasta")

sequence = str(record.seq)

print("Protein:", record.id)

# =========================
# LOAD MODEL
# =========================

print("Loading ESMFold model...")

model = EsmForProteinFolding.from_pretrained(
    "facebook/esmfold_v1",
    cache_dir="./models"
)
tokenizer = AutoTokenizer.from_pretrained(
    "facebook/esmfold_v1",
    cache_dir="./models"
)

model = model.eval()

# CPU
model = model.cpu()

# =========================
# TOKENIZE
# =========================

# tokenizer = model.tokenizer

inputs = tokenizer(
    [sequence],
    return_tensors="pt",
    add_special_tokens=False
)

# =========================
# PREDICT STRUCTURE
# =========================

print("Predicting structure...")

with torch.no_grad():

    output = model(**inputs)

# =========================
# CONVERT TO PDB
# =========================

final_atom_positions = atom14_to_atom37(
    output["positions"][-1],
    output
)

output = {k: v.to("cpu") for k, v in output.items()}

final_atom_positions = final_atom_positions.cpu().numpy()

pdbs = []

for i in range(len(output["aatype"])):

    pdb = to_pdb(
        {
            "aatype": output["aatype"][i],
            "atom_positions": final_atom_positions[i],
            "atom_mask": output["atom37_atom_exists"][i],
            "residue_index": output["residue_index"][i] + 1,
            "b_factors": output["plddt"][i],
        }
    )

    pdbs.append(pdb)

# =========================
# SAVE
# =========================

outfile = f"output/{record.id}.pdb"

with open(outfile, "w") as f:
    f.write(pdbs[0])

print("Saved:", outfile)
