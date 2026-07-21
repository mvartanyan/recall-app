# Packaging Recall for macOS

## Current status

Recall can produce a self-contained Apple-silicon application and DMG. The
package contains the native executable, the bundled desktop view, the local
WeSpeaker ECAPA model, the app icon, Recall's MIT and Apache-2.0 license files,
and the model/dependency notice. It does not contain API keys, conversations,
or other user data.

The current local package is ad-hoc signed. It is useful for build and internal
developer testing, but it is not a public release: Gatekeeper will not trust it
on another Mac. A build intended for ordinary users must be signed with a
Developer ID Application certificate and notarized by Apple.

Current package contract:

- Apple silicon (`arm64`) only
- declared minimum macOS version: 11.0
- hardened runtime enabled by the signing step
- microphone entitlement included
- manual DMG installation and replacement; no auto-updater yet
- bundle identifier remains `com.example.recall` to preserve existing local data

The package has only been run on the current development Mac. Declaring macOS
11.0 as the binary minimum is not the same as testing every supported macOS
release.

## Build an internal test DMG

Install the project dependencies once, then run the local packaging recipe:

~~~sh
cd /Users/michael/dev/recall/app
npm ci
npm run package:mac:local
~~~

The recipe first runs the license and redacted secret-history gates, then builds
an optimized release, ad-hoc signs it, creates the DMG without Finder UI
automation, and runs the package verifier. Outputs are written to:

~~~text
src-tauri/target/release/bundle/macos/Recall.app
src-tauri/target/release/bundle/dmg/Recall_<version>_aarch64.dmg
~~~

Run the verifier again without rebuilding:

~~~sh
npm run verify:mac:package
~~~

It checks code-signature integrity, the microphone entitlement, the declared
minimum macOS version, Apple-silicon code, the pinned speaker-model checksum,
the bundled source-license files and attribution notice, the presence of a DMG,
and the absence of API keys or a Recall database inside the app.

Run publication checks against the lockfiles and Git history before packaging:

~~~sh
npm run audit:licenses
npm run audit:secrets
~~~

Do not publish the output of `package:mac:local` as a normal end-user release.
It deliberately uses the ad-hoc signing identity `-` and is expected to fail a
Gatekeeper assessment after download quarantine is applied.

## Produce a trusted external release

A normal outside-the-Mac-App-Store release requires:

1. Apple Developer Program membership.
2. A **Developer ID Application** certificate installed in the build keychain.
3. App Store Connect API credentials, or an Apple ID app-specific password, for
   notarization.
4. A final reverse-DNS bundle identifier and an explicit migration from
   `~/Library/Application Support/com.example.recall` if that identifier changes.
5. A decision on Apple-silicon-only versus Intel/universal distribution.
6. Regenerated and manually reviewed third-party dependency notices for the
   exact release lockfiles. Recall's own source-license decision is complete:
   users may choose MIT or Apache-2.0.
7. Acceptance testing on every macOS release that will be advertised as supported.

Tauri reads signing and notarization credentials from environment variables.
With App Store Connect API credentials, the release build shape is:

~~~sh
cd /Users/michael/dev/recall/app
export APPLE_SIGNING_IDENTITY="Developer ID Application: ORGANIZATION (TEAMID)"
export APPLE_API_ISSUER="APP_STORE_CONNECT_ISSUER_ID"
export APPLE_API_KEY="APP_STORE_CONNECT_KEY_ID"
export APPLE_API_KEY_PATH="/absolute/path/to/AuthKey_KEY_ID.p8"
CI=true npm run build -- --bundles app,dmg
npm run verify:mac:package
~~~

Never commit the certificate, private key, notarization key, Soniox key, or
OpenAI key. CI secrets are appropriate once release automation is added.

Before publishing, verify the exact downloaded artifact on a clean Mac:

~~~sh
codesign --verify --deep --strict --verbose=2 /Applications/Recall.app
spctl --assess --type execute --verbose=4 /Applications/Recall.app
xcrun stapler validate /Applications/Recall.app
~~~

Then exercise first launch, microphone permission, Soniox key setup, live
captions, final transcription, local voice matching, app restart/persistence,
and optional OpenAI recap. Test installation by dragging Recall into
`/Applications`; launching a bundle directly from the mounted DMG is not the
supported installed state.

## Architecture choice

The verified artifact is `arm64`, because this development machine currently
has only the `aarch64-apple-darwin` Rust target. A universal build would require
the Intel target and a successful build of every native dependency for both
architectures. The candidate Tauri target is `universal-apple-darwin`, but
sherpa-onnx/CoreAudio packaging must be proven before documenting that as a
working release recipe. The practical first beta is Apple silicon only; add an
Intel or universal artifact only if there are actual Intel testers.

## Updates and compatibility

There is no updater configured. The initial distribution model is a versioned
DMG: quit Recall, replace the application in `/Applications`, and reopen it.
The local database and keys remain under the bundle identifier's Application
Support directory and are not part of the app bundle.

Before changing `com.example.recall`, implement and test a one-time data
migration. Changing the identifier without migration makes the new app look
empty and leaves the user's existing conversations and credentials behind in
the old directory.

## Product readiness outside packaging

Packaging does not remove two current beta constraints:

- Recall records one CoreAudio input. Capturing both microphone and remote
  participants currently requires a virtual or aggregate device such as
  BlackHole; native ScreenCaptureKit system-audio capture is pending.
- Cross-meeting ECAPA voice matching is intentionally conservative and still
  needs labelled multi-speaker calibration before it should be described as
  generally reliable.

Relevant upstream guidance:

- https://v2.tauri.app/distribute/dmg/
- https://v2.tauri.app/distribute/sign/macos/
- https://developer.apple.com/developer-id/
- https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution
