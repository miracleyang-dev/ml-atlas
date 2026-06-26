#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# fetch-assets.sh
#
# Downloads KaTeX (CSS/JS + math fonts) and the Inter / JetBrains Mono
# woff2 subsets into docs/assets/. These directories are git-ignored and
# regenerated on every build, so the site has zero runtime dependency on
# fonts.googleapis.com / fonts.gstatic.com / cdn.jsdelivr.net — useful for
# users on restricted networks (e.g. mainland China without VPN).
#
# Run from anywhere; paths are resolved relative to this script.
# ----------------------------------------------------------------------------
set -euo pipefail

# Repo root = parent of docs/
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ASSETS_DIR="$DOCS_DIR/assets"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Pinned versions — bump deliberately when KaTeX / fontsource releases change.
KATEX_VER="0.16.11"
INTER_VER="5.1.0"
JBM_VER="5.1.0"

NPM_REGISTRY="${NPM_REGISTRY:-https://registry.npmjs.org}"

# Helper: download + extract an npm tarball into $TMP/<name>/package/
fetch_npm() {
  local pkg="$1" ver="$2" name="$3" url
  if [[ "$pkg" == @* ]]; then
    local scope="${pkg%%/*}"
    local short="${pkg##*/}"
    url="$NPM_REGISTRY/$scope/$short/-/$short-$ver.tgz"
  else
    url="$NPM_REGISTRY/$pkg/-/$pkg-$ver.tgz"
  fi
  echo "  -> $url"
  curl -fsSL "$url" -o "$TMP/$name.tgz"
  mkdir -p "$TMP/$name"
  tar -xzf "$TMP/$name.tgz" -C "$TMP/$name"
}

# --- KaTeX -------------------------------------------------------------------
echo "[fetch-assets] KaTeX $KATEX_VER"
fetch_npm "katex" "$KATEX_VER" "katex"
rm -rf "$ASSETS_DIR/katex"
mkdir -p "$ASSETS_DIR/katex"
cp -R "$TMP/katex/package/dist/." "$ASSETS_DIR/katex/"
# Strip non-runtime files the npm tarball ships in dist/ (README.md, etc.).
# Without this, mkdocs --strict aborts because the markdown isn't in nav.
find "$ASSETS_DIR/katex" -type f \
  \( -iname "*.md" -o -iname "*.txt" -o -iname "LICENSE*" \
     -o -iname "CHANGELOG*" -o -iname "CONTRIBUTING*" \) \
  -delete

# --- Inter (latin subset, weights 400/500/600/700) ---------------------------
echo "[fetch-assets] @fontsource/inter $INTER_VER"
fetch_npm "@fontsource/inter" "$INTER_VER" "inter"
mkdir -p "$ASSETS_DIR/fonts/inter"
for w in 400 500 600 700; do
  src="$TMP/inter/package/files/inter-latin-${w}-normal.woff2"
  if [[ -f "$src" ]]; then
    cp "$src" "$ASSETS_DIR/fonts/inter/"
  else
    echo "  ! missing: inter-latin-${w}-normal.woff2 (skipped)"
  fi
done

# --- JetBrains Mono (latin subset, weights 400/700) --------------------------
echo "[fetch-assets] @fontsource/jetbrains-mono $JBM_VER"
fetch_npm "@fontsource/jetbrains-mono" "$JBM_VER" "jbm"
mkdir -p "$ASSETS_DIR/fonts/jetbrains-mono"
for w in 400 700; do
  src="$TMP/jbm/package/files/jetbrains-mono-latin-${w}-normal.woff2"
  if [[ -f "$src" ]]; then
    cp "$src" "$ASSETS_DIR/fonts/jetbrains-mono/"
  else
    echo "  ! missing: jetbrains-mono-latin-${w}-normal.woff2 (skipped)"
  fi
done

echo "[fetch-assets] Done. Assets in: $ASSETS_DIR"
