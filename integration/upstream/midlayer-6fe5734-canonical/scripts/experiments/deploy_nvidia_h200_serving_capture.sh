#!/bin/bash
# Usage: deploy_nvidia_h200_serving_capture.sh  (from a clean checkout; prints the snapshot dir)
# Deploy the committed worktree snapshot (LFS-smudged files) to h200 /mnt/public.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ -n "$(git status --porcelain)" ]; then echo "worktree dirty" >&2; exit 1; fi
SHA=$(git rev-parse HEAD); SHORT=${SHA:0:9}
DEST=/mnt/public/yangxinhao/src/midlayer_h200_serving_capture/$SHORT
git ls-files -z | tar --null -T - -czf /tmp/midlayer_$SHORT.tgz
timeout 600 scp -q /tmp/midlayer_$SHORT.tgz h200-tmp:/mnt/public/yangxinhao/src/
timeout 300 ssh h200-tmp "set -e; mkdir -p $DEST && cd $DEST && tar xzf /mnt/public/yangxinhao/src/midlayer_$SHORT.tgz && echo $SHA > SOURCE_COMMIT && mkdir -p build && cc -O3 -shared -fPIC -o build/lru_sim.so src/kernel/csrc/lru_sim.c && cd /mnt/public/yangxinhao/src && rm midlayer_$SHORT.tgz && ln -sfn $DEST /mnt/public/yangxinhao/src/midlayer_h200_serving_capture/current"
(cd /tmp && rm "midlayer_$SHORT.tgz")
echo $DEST
