// Keep only the latest parameter set, even if an older request ignores cancellation.
export function createLinkPredictionRunner({
  predict, onState, delayMs = 300, timeoutMs = 10000,
}) {
  let revision = 0;
  let timer;
  let controller;

  function cancel() {
    revision += 1;
    clearTimeout(timer);
    controller?.abort();
    controller = undefined;
  }

  function schedule(payload) {
    cancel();
    const currentRevision = revision;
    onState({ status: "loading" });
    timer = setTimeout(async () => {
      const currentController = new AbortController();
      controller = currentController;
      const timeout = setTimeout(() => currentController.abort(), timeoutMs);
      try {
        const result = await predict(payload, { signal: currentController.signal });
        if (revision === currentRevision) onState({ status: "ready", result });
      } catch (error) {
        if (revision === currentRevision) {
          onState({
            status: "error",
            message: currentController.signal.aborted
              ? "预测超时，请重试"
              : error.message || "带宽预测失败，请重试",
          });
        }
      } finally {
        clearTimeout(timeout);
        if (controller === currentController) controller = undefined;
      }
    }, delayMs);
  }

  return { schedule, cancel };
}
