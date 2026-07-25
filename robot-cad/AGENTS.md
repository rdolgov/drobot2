# Robot CAD agent contract

Use the installed `cad:cad`, `cad:cad-viewer`, and `cad:step-parts` skills for
CAD work in this repository.

- Treat Python generators and YAML specifications as the editable source.
- Treat STEP as the primary generated artifact; STL/3MF/GLB are secondary.
- Generate explicit targets only through the CAD skill's STEP workflow.
- After geometry changes, inspect refs/facts/planes/positioning and run targeted
  measurements for the affected fit or datum.
- Always create and visually review a snapshot of each changed primary STEP.
- Always hand changed CAD artifacts to CAD Viewer and return live review links.
- Use CAD Viewer's Orthographic markup workspace for visual change requests.
- Launch CAD Viewer from the repository-owned markup runtime under
  `tools/cad-viewer-markup/runtime/`. Do not depend on or fall back to a
  neighboring `text-to-cad` checkout.
- Save editable markup JSON and exported PNG review sheets under `reviews/`.
- Interpret markup colors as red=remove, green=add, blue=move,
  purple=hardware, and amber=note.
- Keep `vendor/` models immutable. Search step.parts before adding or faking a
  purchasable component, and record provenance plus SHA-256 in
  `vendor/README.md`.
- Preserve the ST3215 motor-bay cavity code as the single fit-critical source.
  Other parts must import and place it rather than copy its cavity geometry.
- Do not silently substitute ST3215 for STS3212 or ST3215-HS. Require an exact,
  verified STEP before changing the configured servo variant.
- Generated files under `exports/` are disposable and should not be committed.

The original SO-101 STEP coordinate system is intentionally preserved. Read
`specs/coordinate-system.md` and the relevant YAML specification before
changing geometry.
