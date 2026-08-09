# Orthographic CAD markup integration

This folder makes the custom CAD Viewer markup workspace part of the robot
project itself.

- Source repository: `earthtojake/text-to-cad`
- Source commit: `86a39ba Add orthographic CAD markup workspace`
- Migrated patch: `0001-Add-orthographic-CAD-markup-workspace.patch`
- Project runtime: `runtime/`
- Runtime identity: `0.3.9-drobot2-markup.1`

`runtime/` contains the packaged CAD Viewer backend and support packages from
the installed CAD 0.3.9 plugin, with the frontend rebuilt from the source tree
containing the markup commit. It is committed to this repository and requires
neither a neighboring `text-to-cad` checkout nor its `node_modules`.

`scripts/start_cad_viewer.ps1` launches this runtime directly and reuses only a
Viewer reporting the same project runtime identity. The patch remains the
source-level record for review, rebasing, and future Viewer upgrades.

To recover the feature in a compatible clean `text-to-cad` checkout:

```powershell
git switch develop
git apply --check <path-to-cad>/tools/cad-viewer-markup/0001-Add-orthographic-CAD-markup-workspace.patch
git apply <path-to-cad>/tools/cad-viewer-markup/0001-Add-orthographic-CAD-markup-workspace.patch
npm --prefix viewer install
npm --prefix viewer run test
```

After applying the patch to a newer compatible Viewer, run its tests and build,
then refresh `runtime/` from the packaged Viewer backend plus the newly built
`dist/`. Review the patch before upgrading because UI integration points may
have changed.
