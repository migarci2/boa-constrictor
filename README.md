# BoaConstrictor: A Mamba-based Lossless Compressor for High Energy Physics Data [![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

This repo provides a byte-level compression pipeline driven by a neural predictor (BoaConstrictor) and entropy coding (range coding). It includes:

- A clean CLI to train a model, compress with it, and decompress back
- Per-experiment YAML configs and an interactive config creator
- Optional progress bars and timing for each major stage
- CPU and GPU execution, with tips for best performance

Key entrypoints:
- CLI: `main.py`
- Example config: `experiments/cms_experiment/cms_experiment.yaml`

> [!NOTE]  
> **Reference implementation for GPU Portability**  
> The `portability_solved_cpp` folder contains a reference implementation of BOA using the Mamba network in C++. This implementation specifically solves portability issues on GPUs for CUDA. Please note that it includes only compression/decompression logic and does not contain code for training.


## Quick start

1) Install dependencies (PyTorch not pinned here; use the build suited for your system):

```bash
python3 -m pip install -r requirements.txt
```

2) Create a config interactively and run the experiment:

```bash
python3 main.py --new-experiment
```

3) Or run with an existing config and show timings:

```bash
python3 main.py --config experiment_name --show-timings
```

Useful flags:
- `--no-progress` to disable progress bars
- `--device cpu|cuda` to override device
- `--precision fp32|fp16|fp8` to override compute precision (training only)
- `--train-only`, `--compress-only`, `--decompress-only` to run specific stages
- `--model-path /path/to/model.pt` to load a pre-trained checkpoint and skip training (also supported via `model_path` in the YAML)
- `--verify` to verify the files after compression-decompression cycle
- `--evaluate`, `--evaluate-only` to evaluate performance of the compression model
- `--comparison-baseline-only` to run LZMA and ZLIB on the dataset as baselines

> [!WARNING]  
Currently training can only be done on a CUDA-Compatible GPU!

## Config file structure

A minimal example (`configs/experiment.yaml`):

```yaml
name: example_experiment
file_path: /path/to/dataset.bin
progress: true
device: cuda
precision: fp16

# Optional: set a checkpoint to skip training
# Path can be absolute or relative to this YAML file
# model_path: /path/to/checkpoints/example_experiment_final_model_fp16.pt

dataloader:
  seq_len: 32768
  batch_size: 3

model:
  backbone: mamba  # or mingru for the recurrent PoC
  d_model: 256
  num_layers: 8

training:
  lr: 5e-4
  epochs: 50

compression:
  chunks_count: 1000
  file_to_compress: ''

benchmark:
  quantization_bits: []  # example: [8, 4] for weight-only post-training benchmarks

splits: [0.8, 0.1, 0.1]
```

- `file_path` should point to the raw bytes file to train/encode.
- `splits` should sum to 1.0; if not, defaults are applied.
- `chunks_count` controls how many chunks are used during compression; see Performance notes below.
- `model.backbone: mingru` swaps the predictor to the new recurrent PoC while keeping the same BOA/range-coder pipeline.
- `benchmark.quantization_bits` runs extra weight-only low-bit roundtrips and writes a JSON report in the experiment directory.


## CLI overview

`main.py` wires together:
- Reading input bytes
- Building the model (`BoaConstrictor`) and `ByteDataloader`
- Splitting into train/val/test (`make_splits`)
- Training via `train(...)`
- Compression/Decompression via `BoaFile.compress(...)` / `BoaFile.decompress(...)`

Timings are printed when `--show-timings` is used. Progress bars respect `progress: true` (in config) unless `--no-progress` is passed.

## Quick Colab benchmark

If you want the shortest path in Google Colab, use [`colab_benchmark.py`](/home/dark/Desktop/Projects/boa-constrictor/colab_benchmark.py). It writes a config for you and launches `main.py` with `mingru`, verification, timings, and optional low-bit variants.

```bash
git clone <your-repo-url>
cd boa-constrictor
python -m pip install -r requirements.txt

# Put your binary at /content/data.bin first
python colab_benchmark.py \
  --data /content/data.bin \
  --name mingru_colab \
  --backbone mingru \
  --epochs 5 \
  --seq-len 4096 \
  --batch-size 4 \
  --quant-bits 8,4
```

Useful overrides:
- `--gpu-streams 256` if Colab VRAM is tight.
- `--backbone mamba` to compare against the original model.
- `--extra-main-args "--evaluate"` if you also want evaluation plots.
- `--no-run` to only generate the YAML and inspect it first.

Outputs:
- `experiments/<name>/<name>.boa`
- `experiments/<name>/<name>_benchmark_report.json`
- `experiments/<name>/quantized_benchmarks/` for low-bit variants


## Architecture and data flow

1) Byte modeling (neural predictor)
   - The `BoaConstrictor` model receives byte sequences and predicts a distribution over the next byte (0..255) at each position.
   - Training minimizes cross-entropy between predictions and observed bytes.

2) Entropy coding (range coding)
   - For each byte to be stored, the predictor provides probabilities p(b | context).
   - A range coder converts these probabilities and symbols into a compact bitstream close to the theoretical entropy (−log₂ p).

3) Container and chunks
   - Data is processed in chunks, enabling parallelism and streaming.
   - Each chunk stores (a) first bytes, (b) the compressed range-coded stream, and (c) metadata.

4) Decompression mirrors compression
   - The range decoder reconstructs each symbol using the same probabilities generated by the model conditioned on previously decoded bytes (and chunk state).


## Range coding primer (entropy coding)

Range coding is a practical form of arithmetic coding. At a high level, it maintains an interval [low, high) within [0, 1) representing the current coder state. For each symbol with probability distribution {p_i} over the alphabet:

- Partition the current interval into sub-intervals proportional to {p_i}
- Select the sub-interval for the observed symbol
- Renormalize when the interval becomes too small, emitting bits

Conceptually, after encoding a sequence x₁…x_T, the final interval size is approximately Π_t p(x_t | context), so the total code length approaches −Σ_t log₂ p(x_t | context) bits.

A simplified encode step with cumulative frequencies (integer-scaled):

```
state: low=0, high=RANGE_MAX
for symbol s with cumulative counts C and total T:
  range = high - low + 1
  high  = low + (range * C[s+1] // T) - 1
  low   = low + (range * C[s]   // T)
  while renormalization_condition(low, high):
    output_bit_and_shift(low, high)
```

- `C[k]` is the cumulative count of symbols < k (C[0] = 0, C[Σ] = T)
- Renormalization shifts out stable MSBs so the internal registers don’t overflow
- Decoding performs the inverse using the same `C` and `T`

Why range coding here?
- It’s simple, fast, and numerically stable vs naive arithmetic coding
- Integer arithmetic avoids floating-point drift
- It compresses close to the entropy bound, assuming good probability estimates from the model


## CPU and GPU performance notes

Compression/decompression performance hinges on two main costs:
1) Probability computation (neural model inference)
2) Range coder symbol processing

Range coder serialism vs parallelism
- The range coder is intrinsically sequential per symbol. However, you can parallelize across independent chunks.
- Choose `compression.chunks_count` to balance parallelism and overhead. Too many tiny chunks increase metadata and launch overhead; too few large chunks underutilize parallel resources.

CPU speedups
- Vectorized preprocessing: Prefer NumPy or PyTorch tensor operations on large slices over Python loops.
- Chunk sizing: Tune `chunks_count` so each chunk fits in cache and reduces memory stalls.
- Threaded inference: If running on CPU, enable MKL/OpenMP threading for BLAS (subject to your PyTorch build). Typical knobs: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `torch.set_num_threads`.
- I/O buffering: Read the dataset once into memory when feasible; use memory-mapped I/O for very large files.

GPU speedups
- Batch inference: Evaluate the model on multiple sequences (or longer sequences) in one pass to utilize SMs better.
- Mixed precision: Use `precision: fp16` (or `--precision fp16`) to halve bandwidth and often speed up GEMMs/attention layers on supported GPUs (training only).
- Chunk-level parallelism: Schedule multiple chunks concurrently so the GPU is fed continuously; avoid tiny chunks that cause excessive kernel launch overhead.
- Custom GPU range coder for batch independent compression

### Streaming compression batches

Compression runs in batches of chunks to keep memory usage bounded. By default, the batch size ("gpu_streams") is chosen automatically based on your configuration. For demos or reproducibility, you can force a fixed batch size via an environment variable:

```bash
# Example: process 10,000 chunks in two streaming batches of 5,000 each
export BOA_GPU_STREAMS=5000
python3 main.py --config your_experiment
```

With `chunks_count: 10000` (or when the input produces 10,000 chunks), this will compress in two waves of 5,000 chunks each, demonstrating the streaming pattern (write-as-you-go with an index finalized at the end).

## Reproducibility and checkpoints

- The CLI saves/loads model checkpoints according to the `train(...)` implementation. Keep names consistent with your `name` field so compression uses the trained model you expect.
- For long runs, prefer deterministic flags where feasible (e.g., set random seeds) but note that GPU determinism can reduce performance and is not always guaranteed. Only guaranteed determinism is across **same** software stack and **same** hardware stack.


## Troubleshooting

- `file_path` not found:
  - Update the YAML to point to an existing dataset file. For a smoke test, use a small file first.
- CUDA out of memory:
  - Reduce `batch_size`, decrease `seq_len`. Ensure other processes aren’t using VRAM.
- Slow throughput on GPU:
  - Increase chunk-level parallelism and batch size, and avoid tiny chunks.


## References and further reading
- Range coding (arithmetic coding): classic papers and tutorials provide in-depth renormalization details and proofs.
- Neural compression literature for modeling bytes/sequences with transformers and state-space models (e.g., Mamba).

## Citation
If you use this codebase, or otherwise find our work valuable, please cite BOA Constrictor:
```
@misc{gupta2025boaconstrictormambabasedlossless,
      title={BOA Constrictor: A Mamba-based lossless compressor for High Energy Physics data}, 
      author={Akshat Gupta and Caterina Doglioni and Thomas Joseph Elliott},
      year={2025},
      eprint={2511.11337},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/2511.11337}, 
}
@software{gupta_2025_17571973,
  author       = {Gupta, Akshat and
                  Doglioni, Caterina and
                  Elliott, Thomas},
  title        = {Boa Constrictor: A Mamba-based Lossless Compressor
                   for High Energy Physics data
                  },
  month        = nov,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.17571973},
  url          = {https://doi.org/10.5281/zenodo.17571973},
  swhid        = {swh:1:dir:7273b2950222286fe7622e7c545a5806863d1afa
                   ;origin=https://doi.org/10.5281/zenodo.17571972;vi
                   sit=swh:1:snp:6b782111318d9521b182d6fab427ad97d9ea
                   17ad;anchor=swh:1:rel:355c1a3afc7bb7536829745e9c53
                   0fe831265922;path=boa-constrictor-1.0.0
                  },
}
```
## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**.

See the [LICENSE](LICENSE) file for details.

## Disclosure of Delegation to Generative AI

The authors declare the use of generative AI in the research and writing process. According to the GAIDeT taxonomy (2025), the following tasks were delegated to GAI tools under **full** human supervision:

- Feasibility assessment and risk evaluation
- Preliminary hypothesis testing
- Evaluation of the novelty of the research and identification of gaps
- Code generation
- Code optimisation
- Creation of algorithms for data analysis
- Visualization
- Proofreading and editing
- Summarising text
- Adapting and adjusting emotional tone
- Reformatting
- Preparation of press releases and outreach materials
- Quality assessment

The GAI tool used were: ChatGPT-5, Gemini 2.5 Pro, Claude Sonnet 4.5.
Responsibility for the final manuscript lies entirely with the authors.
GAI tools are not listed as authors and do not bear responsibility for the final outcomes.
Declaration submitted by: Akshat Gupta
