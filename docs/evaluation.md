# Evaluation Notes: Brain-Tumor-Segmentation (breast ultrasound)

## Metrics

- **Segmentation quality**: Dice coefficient / F-score and IoU (Jaccard),
  matching the losses/metrics the model is compiled with in
  `src/model.py::build_unet` (`segmentation_models.losses.DiceLoss`,
  `metrics.IOUScore`, `metrics.FScore`). No accuracy/threshold target is
  fixed yet, and no Dice/IoU number is reported below -- the model is built
  with `encoder_weights=None` and there is no training run against the real
  dataset in this repo, so a computed Dice/IoU here would not reflect real
  segmentation quality. The README's Example Output section instead shows
  the pipeline's actual predicted-mask output on the synthetic sample
  image, generated via `scripts/generate_result_preview.py`.
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
| 2026-09-20 | `pytest tests/` (Python 3.10, full stack installed) | tests passed | 26 / 26 | Added `src/preview.py` + `scripts/generate_result_preview.py` and checked in `docs/sample_output.png` (scan / ground-truth mask / predicted mask montage) so the README shows a real pipeline result. Untrained (`encoder_weights=None`), no Dice/IoU reported -- see note above. |
