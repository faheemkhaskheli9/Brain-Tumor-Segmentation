# Evaluation Notes: Brain-Tumor-Segmentation (breast ultrasound)

## Metrics

- **Segmentation quality**: Dice coefficient / F-score and IoU (Jaccard),
  matching the losses/metrics the model is compiled with in
  `src/model.py::build_unet` (`segmentation_models.losses.DiceLoss`,
  `metrics.IOUScore`, `metrics.FScore`). No accuracy/threshold target is
  fixed yet -- Phase 1 only establishes the pipeline and a runnable
  entrypoint (see issue #2 for adding a real example output/metric).
- **Pipeline correctness**: `pytest tests/` green is the pass/fail bar for
  the preprocessing pipeline, the model build/forward/train step (skipped,
  not failed, where the pinned TF stack isn't installed), and the CLI's
  I/O path (`tests/test_cli.py`).

## Reproducing Results

```bash
python -m venv .venv && .venv\Scripts\activate   # Python 3.9 or 3.10, see requirements.txt
pip install -r requirements.txt
pytest tests/
python scripts/segment.py   # runs the real model on the sample image
```

## Result Log

| Date | Config | Metric | Value | Notes |
|------|--------|--------|-------|-------|
| 2026-09-20 | `pytest tests/` (Python 3.10, full stack installed) | tests passed | 22 / 22 | Scaffold work: added `src/cli.py` + `scripts/segment.py` CLI entrypoint, `.vscode/launch.json`, sample synthetic assets. No accuracy metric yet -- pipeline/build verification only. |
