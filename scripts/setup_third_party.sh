#!/usr/bin/env bash
# Fetch the pinned third-party code into third_party/ (not tracked in this repo).
#   curobo            v0.7.8 (d64c4b0)  -> build it afterwards, see SETUP_curobo.md
#   mujoco_menagerie  370c439, only universal_robots_ur5e (sparse checkout; the full repo is ~2.5 GB)
# usage: bash scripts/setup_third_party.sh [target_dir]      (default: third_party/ next to this script's parent)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="${1:-$ROOT/third_party}"
CUROBO_TAG=v0.7.8
CUROBO_SHA=d64c4b005459db10c5dd867d8b30a87d5bda9bdb
MENAGERIE_SHA=370c43965323c0944b433b202aac5058b5f907a9
mkdir -p "$TP"

if [ ! -d "$TP/curobo/.git" ]; then
  git clone --branch "$CUROBO_TAG" --depth 1 https://github.com/NVlabs/curobo.git "$TP/curobo"
fi
got=$(git -C "$TP/curobo" rev-parse HEAD)
[ "$got" = "$CUROBO_SHA" ] || echo "warning: curobo is at $got, expected $CUROBO_SHA ($CUROBO_TAG)"

if [ ! -d "$TP/mujoco_menagerie/.git" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git "$TP/mujoco_menagerie"
  git -C "$TP/mujoco_menagerie" sparse-checkout set universal_robots_ur5e
  git -C "$TP/mujoco_menagerie" -c advice.detachedHead=false checkout "$MENAGERIE_SHA"
fi
got=$(git -C "$TP/mujoco_menagerie" rev-parse HEAD)
[ "$got" = "$MENAGERIE_SHA" ] || echo "warning: mujoco_menagerie is at $got, expected $MENAGERIE_SHA"

test -f "$TP/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
echo "third-party code ready in $TP"
