#!/usr/bin/env bash
# 从 Rockchip GitHub 下载并安装 rknn-toolkit2 的 aarch64 wheel。
# PyPI 上只有 x86_64 的 rknn-toolkit2 wheel，板端（aarch64）需手动安装。
#
# 用法（在 RK3588 板端项目根目录）：
#   bash scripts/install_rknn_toolkit2_aarch64.sh
#   RKNN_TOOLKIT2_VERSION=2.3.2 bash scripts/install_rknn_toolkit2_aarch64.sh
#
# 前置：已执行 `uv sync --extra all`，并在当前 venv 中运行。
set -euo pipefail

VERSION="${RKNN_TOOLKIT2_VERSION:-2.3.2}"
REPO="airockchip/rknn-toolkit2"
BASE_URL="https://raw.githubusercontent.com/${REPO}/master/rknn-toolkit2/packages/arm64"

ARCH="$(uname -m)"
if [ "$ARCH" != "aarch64" ]; then
  echo "本脚本仅安装 aarch64 wheel，当前架构: ${ARCH}" >&2
  echo "x86_64 环境请直接: uv sync --extra export（从 PyPI 安装）" >&2
  exit 1
fi

# 探测可用的 Python 解释器（venv 激活时为 python，否则 python3 / uv run python）
PYTHON=""
for cand in python python3; do
  if command -v "$cand" >/dev/null 2>&1; then PYTHON="$cand"; break; fi
done
if [ -z "$PYTHON" ]; then
  if command -v uv >/dev/null 2>&1; then PYTHON="uv run python"; else
    echo "未找到 python / python3 / uv，请先 uv sync --extra all" >&2; exit 1
  fi
fi

PY_TAG="$($PYTHON -c 'import sys; print("cp%d%d" % sys.version_info[:2])')"
case "$PY_TAG" in
  cp310|cp311|cp312) ;;
  *)
    echo "rknn-toolkit2 arm64 wheel 不支持当前 Python: ${PY_TAG}（仅 cp310/cp311/cp312）" >&2
    exit 1
    ;;
esac

WHL="rknn_toolkit2-${VERSION}-${PY_TAG}-${PY_TAG}-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
URL="${BASE_URL}/${WHL}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "下载: ${URL}"
"$PYTHON" - "$URL" "$TMP/$WHL" <<'PY'
import sys, urllib.request
url, dst = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
    f.write(r.read())
print(f"已保存: {dst}")
PY

echo "安装: ${WHL}"
uv pip install "$TMP/$WHL"

echo "完成。验证: python -c 'from rknn.api import RKNN; print(RKNN)'"
