"""Inference worker: cxr_core.preprocess → ONNX Runtime → CAM → POST /internal/results.

Must not import Django or torch. Runs Celery's solo pool, one job at a time per process, so
a model swap triggered by `model.events` always happens between jobs (see app.py).
"""
