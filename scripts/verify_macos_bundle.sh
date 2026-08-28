#!/bin/sh

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(dirname -- "$script_dir")
app_path=${1:-"$project_dir/src-tauri/target/release/bundle/macos/Recall.app"}
info_plist="$app_path/Contents/Info.plist"
executable="$app_path/Contents/MacOS/recall"
bundled_model="$app_path/Contents/Resources/models/spkrec-ecapa-voxceleb.onnx"
source_model="$project_dir/models/spkrec-ecapa-voxceleb.onnx"
bundled_vad="$app_path/Contents/Resources/models/silero_vad.onnx"
source_vad="$project_dir/models/silero_vad.onnx"
notice="$app_path/Contents/Resources/THIRD_PARTY_NOTICES.md"
license_mit="$app_path/Contents/Resources/LICENSE-MIT"
license_apache="$app_path/Contents/Resources/LICENSE-APACHE"
dmg_dir="$project_dir/src-tauri/target/release/bundle/dmg"

fail() {
  echo "package verification failed: $*" >&2
  exit 1
}

[ -d "$app_path" ] || fail "missing app bundle at $app_path"
[ -f "$info_plist" ] || fail "missing Info.plist"
[ -x "$executable" ] || fail "missing native executable"
[ -f "$bundled_model" ] || fail "missing bundled speaker model"
[ -f "$bundled_vad" ] || fail "missing bundled VAD model"
[ -f "$notice" ] || fail "missing bundled third-party notices"
[ -f "$license_mit" ] || fail "missing bundled MIT license"
[ -f "$license_apache" ] || fail "missing bundled Apache-2.0 license"

/usr/bin/codesign --verify --deep --strict --verbose=2 "$app_path"

bundle_id=$(/usr/libexec/PlistBuddy -c "Print :CFBundleIdentifier" "$info_plist")
[ "$bundle_id" = "com.example.recall" ] || fail "unexpected bundle identifier: $bundle_id"

minimum_system=$(/usr/libexec/PlistBuddy -c "Print :LSMinimumSystemVersion" "$info_plist")
[ "$minimum_system" = "11.0" ] || fail "expected macOS 11.0 minimum, found $minimum_system"

architectures=$(/usr/bin/lipo -archs "$executable")
case " $architectures " in
  *" arm64 "*) ;;
  *) fail "Apple silicon architecture is missing: $architectures" ;;
esac

if ! /usr/bin/codesign -d --entitlements :- "$app_path" 2>/dev/null \
  | /usr/bin/grep -q '<key>com.apple.security.device.audio-input</key><true/>'; then
  fail "microphone entitlement is missing"
fi

source_hash=$(/usr/bin/shasum -a 256 "$source_model" | /usr/bin/awk '{print $1}')
bundle_hash=$(/usr/bin/shasum -a 256 "$bundled_model" | /usr/bin/awk '{print $1}')
[ "$source_hash" = "$bundle_hash" ] || fail "bundled speaker model checksum differs from source"
[ "$bundle_hash" = "da11c87ed452e72087beb6f2fe8a2abc0ef722c2f9a641c373678a0917a07e07" ] \
  || fail "speaker model checksum differs from the pinned artifact"

source_vad_hash=$(/usr/bin/shasum -a 256 "$source_vad" | /usr/bin/awk '{print $1}')
bundle_vad_hash=$(/usr/bin/shasum -a 256 "$bundled_vad" | /usr/bin/awk '{print $1}')
[ "$source_vad_hash" = "$bundle_vad_hash" ] || fail "bundled VAD model checksum differs from source"
[ "$bundle_vad_hash" = "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6" ] \
  || fail "VAD model checksum differs from the pinned artifact"

if /usr/bin/find "$app_path" -type f \( -name 'soniox-api-key' -o -name 'openai-api-key' -o -name 'recall.db' \) \
  | /usr/bin/grep -q .; then
  fail "the app bundle contains user data or API credentials"
fi

dmg_count=$(/usr/bin/find "$dmg_dir" -maxdepth 1 -type f -name 'Recall_*.dmg' | /usr/bin/wc -l | /usr/bin/tr -d ' ')
[ "$dmg_count" -ge 1 ] || fail "no Recall DMG found in $dmg_dir"

version=$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$info_plist")
echo "Verified Recall $version ($architectures)"
echo "App: $app_path"
echo "DMG directory: $dmg_dir"
echo "Signature integrity, microphone entitlement, resources, and credential exclusion passed."
