# Environment and verification

The default environment is for Wav2Vec 2.0. It uses PyTorch 2.5.1 and Transformers 4.46.0, matching the original Dockerfile and requirements. The original full dependency list is retained in [provenance](provenance/original-requirements.txt); it mixed multiple CUDA runtimes and unrelated packages, so it is not installed wholesale.

Install using the README. PyTorch wheel alternatives are listed in the [official installation archive](https://pytorch.org/get-started/previous-versions/). The default conda environment uses Python 3.10; the local CPU validation used Python 3.12 on macOS. Neither is a recovered lockfile from the paper run.

## CPU verification

No model download or dataset access is needed for the automated tests. They construct tiny Wav2Vec 2.0 models and synthetic audio locally.

```bash
# Linux CPU environment: install this wheel instead of the CUDA wheel.
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
python -m irhp.train --help
python -m irhp.evaluate --help
```

On macOS, install `torch==2.5.1` from the default package index. CPU training is limited to `--smoke-test --device cpu`. Full experiments target an NVIDIA GPU.

## Docker

```bash
docker build -t irhp .
docker run --rm --gpus all -it \
  -v "$PWD/data:/workspace/IRHP/data" \
  -v "$PWD/outputs:/workspace/IRHP/outputs" \
  irhp bash
```

The NVIDIA Container Toolkit must be installed on the host. The Dockerfile does not install host GPU drivers or the container runtime inside the container. Docker/CUDA execution remains unverified in this cleanup.

## Experiment record

After validating the environment, record the resolved versions and hardware alongside the run:

```bash
python -m pip freeze > outputs/wav2vec2-ls100/requirements-resolved.txt
nvidia-smi > outputs/wav2vec2-ls100/gpu.txt
git rev-parse HEAD > outputs/wav2vec2-ls100/revision.txt
```

The first real training run needs network access to download the pretrained Wav2Vec 2.0 model. Evaluation of an exported checkpoint is local. The supplied checkpoint loader requires the complete export directory, not just the weight file.

## Conformer

Do not install a newer NeMo into the Wav2Vec 2.0 environment without checking dependency compatibility. The original NeMo version and its attention-return patch are missing from the research snapshot. See [Conformer status](conformer.md); no verified installation recipe is claimed for that path yet.
