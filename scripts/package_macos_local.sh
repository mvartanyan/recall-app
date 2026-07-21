#!/bin/sh

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(dirname -- "$script_dir")

cd "$project_dir"

# Refuse to package a tree with unreviewed license identifiers or credential
# patterns in current source/Git patch history.
npm run audit:licenses
npm run audit:secrets

# CI mode avoids Finder automation while constructing the DMG. The ad-hoc
# identity makes this useful for local/internal testing, but it is not a
# substitute for Developer ID signing and Apple notarization.
CI=true APPLE_SIGNING_IDENTITY=- npm run build -- --bundles app,dmg
"$script_dir/verify_macos_bundle.sh"
