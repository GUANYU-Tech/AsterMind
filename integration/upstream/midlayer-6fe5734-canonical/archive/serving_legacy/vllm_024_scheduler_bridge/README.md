# vLLM 0.24 Scheduler bridge toolchain (archived)

Retired on 2026-09-16 from `src/serving/backend/metax_c500_vllm_024_scheduler.py` and `scripts/evaluation/serving/diagnostics/`.

This toolchain ran the real vLLM-metax 0.24.0 `AsyncScheduler` as a JSONL bridge inside the C500 serving container to validate that the shared analytical Scheduler in `src/serving/backend/vllm.py` reproduces the deployment's scheduling decisions. The comparison concluded that Scheduler choice has no material effect on Serving results, so all three hardwares keep using the single shared analytical Scheduler and this validation instrument is archived. It is retained for traceability only; active code must not import or execute it. Retrieve it here if a future vLLM upgrade requires re-validation.
