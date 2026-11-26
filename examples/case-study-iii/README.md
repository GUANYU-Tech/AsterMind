# Case Study III · 600-second busy-window web example

The bundled `inputs.tar.gz` contains offline constellation data, the ground
gateways, the GLM-5.2
deployment, and 1,096 requests from 08:06 to 08:16 UTC in the WildChat-derived
1× trace. This is the busiest 600-second window aligned to a 60-second frame
in that trace. The workload preserves arrival times and input/output lengths.
The scenario configures 35 gateways, of which 21 receive requests in this
window. Trace gateways have already been mapped to these configured sites.
The manifest records the five mapping rules and their request counts.

The example starts with empty KV state and enables prefix caching. The
workload preserves 422 valid parent-turn links within the window. A request
whose prior turn arrived before the window starts with its full prompt.
The Builder starts link assignment at
08:06 using the original orbital elements. The topology has a 60-second
sampling interval and an extra minute to drain responses. The paper's
24-hour 2× replay uses its own continuous topology and session history.

## Install the inputs

Use the Linux environment described in [Build and installation](../../README.md).
Fast uses the pinned vLLM 0.30 CPU scheduler. The 3D viewer also needs the
browser assets built by `scripts/build.sh` and a WebGL 2 browser.

Run the web service from the OrbitServe checkout. It registers the bundled
example on first use. The Builder and simulation run on the machine hosting
the web service.

```bash
bash scripts/run.sh
```

The installer verifies the input digests and registers the topology and
experiment under `viz/.runtime/`. To register just the inputs, run
`.venv/bin/python scripts/install-case-study-iii.py --ensure`. Pass
`--base-url` to print the scene URL for a different service address.

## Run in the browser

1. Under **Physical model source** (「物理模型来源」), select **Case Study III
   (600 s simulation)** from **Built-in examples** (「内置示例」) and click
   **Load model** (「加载模型」). Click **Open the Case III Example** (「点击直达Case III示例实验」)
   to open the prepared deployment and requests with the offline topology.
   To configure your own requests, use **Generate Data and Continue** instead.
   You can also open the scene URL printed by the installer. Confirm that the
   physical topology is ready and the request table lists 1,096 requests.
2. Use the imported **Fast Mode** deployment. It uses GLM-5.2 BF16,
   eight H200 GPUs per satellite, TP8 within a satellite, and PP4 across
   four satellites per Cell. Its compute and HBM efficiency settings are
   70% and 96.25%, the batch token budget is 16,384, and each Cell has
   193,904 KV blocks of 16 tokens.
3. Keep the supplied **Nearest** setting for the initial run. Default
   route selection uses the physical data and deployed ranks.
4. Click **Start experiment and view results** (「开始实验并查看结果」).
   In **Request performance** (「请求性能」), check the request
   rows and the gateway TTFT, TPOT and E2E. In **Scene playback**
   (「场景回放」), use **Play** or advance to a network event. Follow an
   upload or response path on the 3D globe. Both views use the run ID
   created by the button.

To use a different model or rank placement, make a new scene from the
physical topology and generate its deployment through the normal editor.

The same installer also reads the 180-second reproduction bundle with
its seven requests, session links and prepared routes. That bundle is useful
for inspecting the effects of session KV reuse.
