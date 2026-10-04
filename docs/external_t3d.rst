T3D ion-temperature evaluation
=============================

``desc.external.t3d`` evaluates native T3D with an explicitly supplied AI_GX
transport template and runtime. Like ``desc.external.gx.gx``, ``t3d`` accepts a
sequence of current in-memory DESC equilibria and evaluates them in input order.
It returns one ``IonTemperatureProfile`` per equilibrium. The full equilibrium
is saved as native DESC HDF5 and handed to T3D. It defines no scalar objective,
derivative, optimizer, core extrapolation or equilibrium pressure feedback.

Implementation
--------------

``desc/external/t3d.py`` contains configuration, equilibrium handoff, execution,
cache/failure handling, native output parsing and the public profile API.
``desc/external/_t3d_worker.py`` is only the subprocess entry point for the
caller's external Python environment; no private implementation package is used.
Input validation and asset hashing are shared, and archive profile data and
solver metadata are read in one NetCDF context.

Configuration and dependencies
------------------------------

The caller supplies the physics template, trained weights, model paths, external
T3D installation and execution environment. DESC bundles none of these resources
and never downloads models or submits scheduler jobs. The optional ``t3d`` extra
provides NetCDF4 for output parsing and tomli for Python 3.10; TOML uses the
standard library on Python 3.11+. The selected worker interpreter also needs
native T3D/AI_GX, its Torch/JAX dependencies and a compatible DESC installation.

The verified native runtime uses the ``NNITGProxy`` geometry machinery on the
``rmchurch/ti-opt`` DESC branch. AI_GX format/runtime compatibility is checked
through actual source/library and model-checkpoint provenance; the wrapper does
not substitute a different flux model. The initial supported physics mapping is
one hydrogen/H and one electron/e, fixed density and electron temperature,
torflux coordinates and a fixed DESC equilibrium. Other mappings require
separate native verification and are rejected.

Example within an independently authorized execution environment::

    from desc.external.t3d import T3DAdapter, T3DConfig, t3d

    adapter = T3DAdapter(
        T3DConfig(
            template="transport/ai_gx.in",
            python="/path/to/t3d-environment/bin/python",
            source_root="/path/to/source-containing-t3d-package",
            mode="evolve",
            timeout_seconds=600,
        ),
        output_dir="trial/transport-runs",
    )
    ti = t3d([eq], adapter=adapter)[0]
    print(ti.rho, ti.temperature, ti.temperature_units)
    print(ti.species_type, ti.species_tag, ti.time_seconds, ti.status)

``evaluate_t3d(eq, adapter=adapter)`` returns a single profile. Device visibility
and runtime overrides are explicit ``T3DConfig.environment`` settings.
``require_gpu=True`` requires a working single visible GPU, enables DESC's GPU
device before importing its backend, and rejects CPU fallback. Both GPU and CPU
metadata backends must remain available in the worker's JAX configuration.
``mode="initialize"`` constructs native engine/models/geometry without advancing
transport; it cannot return an evolved temperature profile.

Returned profile and evidence
-----------------------------

The profile contains owned read-only Ti and rho arrays, units, species type/tag
and bulk-ion identity, radial coordinate definition, final time in ``[t_ref]``
and seconds, its time normalization and recorded transport step. For torflux,
``rho=sqrt(toroidal_flux/toroidal_flux_LCFS)``. An innermost cell at positive rho
does not represent the magnetic axis; ``includes_magnetic_axis`` is explicit.
No profile interpolation, species average or core scalar is supplied.

``evidence`` retains achieved/requested steps and time, the native stopping
reason, process and evolution completion, and raw native RMS/iteration/time/step
rows with unchanged explicit time settings. A successful requested horizon is
``status="evolved"``. It does not establish scientific convergence:
``scientific_convergence_assessed=False`` and ``scientific_convergence=None``.
No new convergence criterion is introduced. Native T3D appends a final profile
row with previously computed fluxes; final temperature and preceding flux times
are kept distinct. ``to_dict()`` produces an independent JSON-compatible copy.

Provenance records equilibrium/configuration identities, every checkpoint hash,
dirty execution sources, libraries, rendered input/output hashes, native backend
evidence and the original manifest path/hash. Template rendering changes resource
paths only and audits the unchanged physics inputs.

Caching, failures and archived results
-------------------------------------

Atomic run claims and content identities reuse only verified successful results.
Failed, early-stop, timed-out, running or interrupted claims never retry
automatically; an explicit new ``retry_token`` creates a distinct run identity.
Initialize/failed/partial/early-stop results cannot return usable Ti.
``T3DEvaluationError`` retains the run identity/path, original status and available
outcome evidence. Batch failures raise without a partial batch return; previous
successful runs remain cached. All identities, units/shapes, species metadata,
progress and artifact hashes are verified before returning a profile.

Archived output is read without worker execution or a current runtime probe::

    from desc.external.t3d import read_ion_temperature
    ti = read_ion_temperature("trial/transport-runs/<run_id>")

The reader verifies archived hashes and native output against its immutable
input/result. It preserves the historical source/model provenance; current
installation files need not equal that old execution. Live adapter cache lookup
still requires the current configuration/source identity. Implementation changes
therefore create a new live run identity; existing archived results remain
readable without rerunning transport.

Integration boundaries and validation
------------------------------------

Accepted-step cadence and conditional hooks are application responsibilities.
The ``ti_optimization`` project keeps those hooks and its physical template and
model settings separately, importing this reusable evaluator. No hook is
registered in DESC by importing the public module.

Native job 59265356 previously executed one AI_GX transport timestep with all
100 v2 models on one visible A100. The relocated archive reader is validated
against that actual output. A one-step result does not establish convergence.
The native accepted-equilibrium callback recording test and native GPU transport
were validated separately, not as one combined callback-to-transport run.
Relocation validation uses bounded CPU checks and archived native artifacts;
no new GPU run or allocation is requested.

API
---

.. automodule:: desc.external.t3d
   :members:
   :imported-members:
