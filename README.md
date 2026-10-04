# BT5230 Challenge 72 Project

## Student details

- **Name:** S Ayushmaan Patro
- **Roll number:** CO23BTECH11020
- **Institution:** IIT Hyderabad
- **Course:** BT5230 AI in Drug Discovery

## Project overview

This project addresses Challenge 72, **Optical Chemical Structure Recognition for Digitizing Legacy Chemistry Literature**. It converts chemical-structure images into machine-readable SMILES. The main controlled experiment compares MolScribe fine-tuning on clean synthetic molecular depictions with matched fine-tuning on hand-drawn-style augmented depictions. Both selected models are evaluated on all 5,088 images in the DECIMER hand-drawn benchmark using exact SMILES matching and chemistry-aware Tanimoto similarity.

## Reference paper

Rajan et al. (2024), *Advancements in hand-drawn chemical structure recognition through an enhanced DECIMER architecture*, Journal of Cheminformatics. The paper is directly relevant because it studies image-to-SMILES recognition for hand-drawn chemical structures, identifies the difficulty of noisy drawings and stereochemistry, and provides the scientific basis for the external DECIMER comparison used in this project.

- Paper: https://link.springer.com/article/10.1186/s13321-024-00872-7
- Benchmark dataset: https://zenodo.org/records/7617107
- DECIMER software: https://github.com/Kohulan/DECIMER-Image_Transformer

## Methodology

- Curated ChEMBL molecules were split before depiction generation to prevent molecule leakage.
- Clean and hand-drawn-style augmented images used identical molecule partitions and labels.
- Both MolScribe conditions started from the same official checkpoint and used the same 3,758 training molecules, 470 clean validation molecules, seed, optimizer settings, validation set, and one-epoch fine-tuning duration.
- Model files were selected using clean validation data before calculating the DECIMER results.
- Invalid outputs remained in the denominator.
- Evaluation reports valid-SMILES rate, exact isomeric match, exact non-isomeric match, Morgan-fingerprint Tanimoto similarity, paired bootstrap confidence intervals, and chemical-complexity subgroup results.
- The official DECIMER hand-drawn model was evaluated unchanged on a fixed seeded 100-image subset as an external pretrained reference; it is not presented as a project-trained model.

## Headline results

### MolScribe comparison on 5,088 held-out hand-drawn images

| Metric | Clean fine-tuned | Augmented fine-tuned | Augmented minus clean |
|---|---:|---:|---:|
| Valid SMILES | 76.51% | 76.83% | +0.31 percentage points |
| Exact isomeric match | 9.69% | 9.73% | +0.04 percentage points |
| Exact non-isomeric match | 11.16% | 11.05% | -0.12 percentage points |
| Mean Tanimoto | 0.2998 | 0.2941 | -0.0057 |

Validity and exact-match differences were inconclusive. The paired 95% confidence interval for the mean-Tanimoto change was approximately -0.0111 to -0.0005. Therefore, the tested augmentation did not improve the held-out real-image benchmark under this one-epoch transfer design.

### External DECIMER reference on the fixed 100-image sample

| Metric | Official DECIMER hand-drawn model |
|---|---:|
| Valid SMILES | 96.00% |
| Exact isomeric match | 64.00% |
| Exact non-isomeric match | 69.00% |
| Mean Tanimoto | 0.8111 |

## Submission contents

- `src/ocsr_project/`: project source code.
- `scripts/`: dataset, evaluation, analysis, verification, and demonstration commands.
- `tests/`: automated tests.
- `configs/`: experimental configurations.
- `data/manifests/` and `data/raw/`: curated split manifests and source tables.
- `splits/`: exact evaluator-facing train, validation, and test CSV files.
- `artifacts/metrics/`: complete saved metrics and per-image predictions.
- `colab/MolScribe_matched_finetuning.ipynb`: final matched clean/augmented training notebook.
- `colab/MolScribe_single_image_demo.ipynb`: genuine new-image inference demonstration.
- `demo/molscribe_gradio_app.py`: evaluator-facing interface for genuine new-image inference.
- `reports/final/AIDD_Hand_Drawn_OCSR_Thesis.tex`: editable thesis source.
- `reports/final/AIDD_Challenge_72_Final_Thesis.pdf`: compiled thesis PDF.
- `reports/final/AIDD_Challenge72_Presentation.pdf`: presentation PDF.
- `reports/final/AIDD_Challenge72_Presentation.tex`: editable presentation source.
- `requirements.txt`: pinned local verification dependencies.

The remaining course deliverable is the 5-10 minute presentation video or a working video link.

## Environment setup

Run all commands from the project root: the folder containing this README, `pyproject.toml`, and `requirements.txt`. Python 3.11 is recommended and was used for the final clean-environment verification.

Do not create a new environment over an existing `.venv` made with another Python version. Start from a clean submission folder, or remove only the old `.venv` directory before continuing.

On Windows PowerShell, confirm Python 3.11, create the environment, and activate it:

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, use the Python 3.11 executable:

```bash
python3.11 --version
python3.11 -m venv .venv
source .venv/bin/activate
```

After activation, verify that the active interpreter is Python 3.11:

```text
python --version
```

The output must begin with `Python 3.11`. Then install the pinned environment and the project:

```text
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Confirm the evaluation-critical package versions:

```text
python -c "import numpy, pandas, rdkit; print(numpy.__version__, pandas.__version__, rdkit.__version__)"
```

Expected output: `2.3.3 2.3.3 2026.03.6`.

## Exact submitted splits

The scaffold-separated assignments are supplied as three standalone files for direct inspection:

- `splits/train.csv`: 3,758 molecules
- `splits/validation.csv`: 470 molecules
- `splits/test.csv`: 470 molecules

These files are direct subsets of `data/manifests/chembl37_stage_5000.csv` and retain the compound ID, raw SMILES, canonical SMILES, and split label for every molecule.

The pipeline uses `data/manifests/chembl37_stage_5000.csv` as the authoritative split input. That master manifest, with its `split` column, was used to generate the condition-specific manifests and the training packages used in the reported experiments. The three files under `splits/` were exported afterward from those unchanged assignments solely to make each partition easy to inspect and to satisfy the requested submission layout. They do not define a new split, were not substituted retrospectively into the pipeline, and do not change any training or evaluation result. Reproduction commands therefore continue to use the authoritative master manifest.

## Optional full training reproduction

Routine assessment does not require retraining. To regenerate the exact scaffold-separated split from the committed ChEMBL source table, run:

```text
python scripts/prepare_molecules.py data/raw/chembl37_stage_5000.csv data/manifests/chembl37_stage_5000.csv --smiles-column smiles --id-column compound_id --max-tokens 128 --seed 20260915
```

Then recreate the two depiction conditions:

```text
python scripts/generate_images.py data/manifests/chembl37_stage_5000.csv data/manifests/chembl37_stage_5000_clean.csv --condition clean --output-root data/processed/chembl37_stage_5000 --image-size 224 --seed 20260915
python scripts/generate_images.py data/manifests/chembl37_stage_5000.csv data/manifests/chembl37_stage_5000_augmented.csv --condition handdrawn_augmented --output-root data/processed/chembl37_stage_5000 --image-size 224 --seed 20260915
```

Create the clean-training/clean-validation and augmented-training/clean-validation packages:

```text
python scripts/prepare_colab_molscribe.py --manifest data/manifests/chembl37_stage_5000_clean.csv --output artifacts/colab/molscribe_clean_train_validation.zip
python scripts/prepare_colab_molscribe.py --manifest data/manifests/chembl37_stage_5000_clean.csv --training-manifest data/manifests/chembl37_stage_5000_augmented.csv --training-condition handdrawn_augmented --output artifacts/colab/molscribe_augmented_train_clean_validation.zip
```

Upload the selected package to `colab/MolScribe_matched_finetuning.ipynb`. Set its first code cell to `clean` or `augmented` and run each condition in a fresh T4 GPU runtime. The notebook verifies the package, starting checkpoint, repository commit, split counts, and final saved artifacts.

## Recommended professor verification

No GPU, checkpoint download, retraining, or full model inference is needed for routine evaluation. Recompute the complete 5,088-image comparison from the preserved per-image predictions:

```text
python scripts/reproduce_benchmark_metrics.py
```

The output must end with:

```json
"verification": "PASS"
```

Run the automated tests:

```text
python -m pytest -q
```

Expected result: **36 passed**.

To reproduce the fixed external DECIMER comparison from saved results:

```text
python scripts/compare_decimer_external_sample.py
```

## Demonstration and new-input support

For a fast CPU-only replay of a preserved validation image and prediction:

```text
python scripts/run_saved_prediction_demo.py
```

This command writes `artifacts/demo/saved_prediction_demo.png`. It is explicitly a saved-prediction replay and does not perform neural-network inference.

For a genuinely new PNG or JPEG input, use `colab/MolScribe_single_image_demo.ipynb` with a Google Colab T4 GPU and upload `demo/molscribe_gradio_app.py` when prompted. The interface accepts one structure image and returns raw and canonical predicted SMILES, an RDKit validity result, and a rendered molecule when parsing succeeds.

The final checkpoints are available from the following view-only folders:

- Clean fine-tuned checkpoint: https://drive.google.com/drive/folders/11q6gxrcGN4IrUZxhVmS9akKdoz3V9JBx?usp=sharing
- Augmented fine-tuned checkpoint: https://drive.google.com/drive/folders/1tlpDRh4Zg8Hn6QNZAnDDlWLPXcEZimMh?usp=sharing

Expected SHA-256 hashes:

```text
clean:     64a5bf5c356cd535cf0fdc76ef26164bfe00d3ab837c0376d2334ca7e663e3fd
augmented: 20bba8c832365f1b708cfe7738b5dfd5f90b025bf6a4619288240e0c6f313741
```

Training used MolScribe commit `7296a30413eb55436702011efdff78131f66d162` in a pinned Python 3.10.21 Colab environment. Full checkpoint inference and retraining are optional deep-audit paths and are not necessary to verify the reported results.

## Evaluator results guide

The headline tables above and the thesis contain all results needed for normal assessment. The files under `artifacts/metrics/` are retained as machine-readable evidence; an evaluator does not need to inspect every file.

Open these files for the main numerical results:

| Purpose | File to open |
|---|---|
| Complete clean-versus-augmented benchmark comparison | `artifacts/metrics/molscribe/decimer_clean_vs_augmented.json` |
| Clean-model summary on all 5,088 benchmark images | `artifacts/metrics/molscribe/decimer_clean/prediction_decimer_hdm_benchmark_summary.json` |
| Augmented-model summary on all 5,088 benchmark images | `artifacts/metrics/molscribe/decimer_augmented/augmented_prediction_decimer_hdm_benchmark_summary.json` |
| Per-image paired error analysis | `artifacts/metrics/molscribe/decimer_analysis/decimer_paired_error_analysis.csv` |
| Molecular-complexity subgroup results | `artifacts/metrics/molscribe/decimer_analysis/decimer_subgroup_metrics.csv` |
| External DECIMER 100-image reference summary | `artifacts/metrics/decimer_external_reference/decimer_100_image_summary.json` |

File naming convention:

- `*_summary.json`: concise aggregate metrics suitable for direct inspection.
- `*_per_image.csv`: one auditable prediction and score record per image.
- `*_native.csv`: original model output retained for provenance.
- Other top-level metric files record development experiments and data-quality audits; they support the methodology but are not required for reading the final result.

The simplest independent check is still:

```text
python scripts/reproduce_benchmark_metrics.py
```

## Limitations and responsible use

- The synthetic hand-drawn-style augmentation did not close the synthetic-to-real domain gap.
- Recognition declined for more complex, multi-ring, and stereochemically rich structures.
- A valid SMILES is not proof that the predicted molecule matches the input image.
- The system is a research prototype for chemistry-document digitization and must not be used as an unsupervised source of safety-critical chemical information.
- The DECIMER benchmark was not used for training, model selection, or post-result retuning.
