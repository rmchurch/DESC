**Warning:**

These objectives/utilities depend on other codes external to DESC. They are not routinely tested
and may not be compatible with all versions of the codes. The DESC team is not
responsible for maintaining or documenting those external codes, and we do not guarantee
to regularly maintain these objectives/utilities. Use these at your own risk!

The external codes are not included with DESC, and you may need to obtain access to them
to use these objectives/utilities. Those codes may not be publicly available and may require a
license to install them.

`_terpsichore.py` was last tested on September 8, 2025 with DESC v0.15.0 and a version
of TERPSICHORE compiled on Perlmutter in January of 2025.

`paraview.py` was last tested on November 24, 2025 with DESC v0.16.0, Paraview v5.13.3
and pyvista v0.46.4.

``t3d.py`` provides an explicitly configured AI_GX transport evaluator returning
ion-temperature radial profiles with native units and verified provenance.
Templates, trained model assets and scheduler/cadence hooks are supplied by the
application. See ``docs/external_t3d.rst`` for the API and native validation scope.
