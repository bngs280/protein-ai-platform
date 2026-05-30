## use case 1
python run_analysis.py \
    --fasta test_data/test.fasta \
    --mutation R175H

## use case 2
python scripts/esm_analysis.py \
    --fasta test_data/test.fasta \
    --mutation R175H

## use case 3
python scripts/mutation_engine.py \
    --fasta test_data/kras.fasta \
    --mutation G12D

## use case 4
python scripts/structure_predict.py

## use case 5
python scripts/stability_predict.py \
    --fasta test_data/kras.fasta \
    --mutation G12D