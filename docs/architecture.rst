Architecture Overview
=====================

SWE is organized as a collection of loosely coupled subpackages that
form a seismic analysis pipeline.  Each subpackage can be used
independently, but they compose naturally in the order shown below.

Pipeline Workflow
-----------------

.. mermaid::

   graph LR
       IO["io<br/>(readers/writers)"]
       NOISE["noise<br/>(classification)"]
       PICK["picking<br/>(STA/LTA, SeisBench)"]
       ASSOC["association<br/>(phase->event)"]
       DET["detection<br/>(matched-filter)"]
       LOC["location<br/>(NonLinLoc)"]
       VEL["velocity<br/>(travel-time)"]
       MAG["magnitude<br/>(ML, Md)"]
       FOCAL["focal<br/>(mechanisms)"]
       MT["moment_tensor<br/>(decomposition)"]
       STRESS["stress<br/>(inversion)"]
       SPEC["spectral<br/>(coda-Q, ratios)"]
       UNC["uncertainty<br/>(propagation)"]
       VIZ["viz<br/>(plots)"]
       PIPE["pipeline<br/>(YAML orchestration)"]

       IO --> NOISE
       IO --> PICK
       IO --> SPEC
       PICK --> ASSOC
       ASSOC --> DET
       ASSOC --> LOC
       VEL --> LOC
       LOC --> UNC
       UNC --> MAG
       LOC --> MAG
       LOC --> FOCAL
       FOCAL --> MT
       FOCAL --> STRESS
       MAG --> VIZ
       STRESS --> VIZ
       PIPE -.->|orchestrates| IO
       PIPE -.->|orchestrates| PICK
       PIPE -.->|orchestrates| LOC

Data Flow
---------

The typical end-to-end workflow proceeds as follows:

1. **Data ingestion**: ``swe.io`` reads waveforms, station metadata,
   and existing pick/catalog files.
2. **Quality control**: ``swe.noise`` classifies ambient noise and
   computes quality metrics to flag bad time windows.
3. **Phase picking**: ``swe.picking`` produces P- and S-wave arrival
   times using STA/LTA or deep-learning models.
4. **Association**: ``swe.association`` groups picks into candidate
   events.
5. **Detection**: ``swe.detection`` runs matched-filter (template)
   correlation to find additional events.
6. **Location**: ``swe.location`` determines hypocenters, optionally
   using velocity models from ``swe.velocity``.  Optionally,
   ``swe.uncertainty`` propagates pick-time uncertainties through
   Monte Carlo relocation.
7. **Magnitude**: ``swe.magnitude`` estimates local (ML) and duration
   (Md) magnitudes and attaches them to ObsPy Event objects.  When
   location uncertainty is available, magnitude uncertainty includes
   the distance-induced contribution.
8. **Focal mechanisms**: ``swe.focal`` measures P-wave first-motion
   polarities, computes takeoff angles, and solves for focal
   mechanisms via grid search.
9. **Moment tensors**: ``swe.moment_tensor`` decomposes moment
   tensors and computes radiation patterns.
10. **Stress inversion**: ``swe.stress`` inverts clustered focal
    mechanisms for the regional stress tensor.
11. **Visualization**: ``swe.viz`` produces waveform, spectrogram,
    and spectral plots for quality review.

Pipeline Orchestration
----------------------

``swe.pipeline`` provides a YAML-driven orchestration layer that ties
all modules together:

- **DAG execution**: steps are topologically sorted by ``${ref}``
  dependencies and executed in order.
- **Output caching**: step outputs are serialized so unchanged steps
  can be skipped on re-runs.
- **Checkpointing**: interrupted pipelines resume from the last
  successful step via ``swe run --resume latest``.
- **Provenance**: every step records inputs, outputs, timestamps,
  and software versions for reproducibility.
- **Parallel execution**: independent steps run concurrently when
  ``--parallel`` is passed to ``swe run``.

Subpackage Dependencies
-----------------------

.. mermaid::

   graph TD
       subgraph Core
           UTILS["utils<br/>(helpers)"]
       end

       subgraph Pipeline
           IO["io"]
           NOISE["noise"]
           PICK["picking"]
           ASSOC["association"]
           DET["detection"]
           LOC["location"]
           VEL["velocity"]
           MAG["magnitude"]
           FOCAL["focal"]
           MT["moment_tensor"]
           STRESS["stress"]
           SPEC["spectral"]
           UNC["uncertainty"]
           VIZ["viz"]
           TOPO["topo"]
       end

       subgraph Orchestration
           PIPE["pipeline"]
           CLI["cli"]
       end

       UTILS --> IO
       UTILS --> NOISE
       UTILS --> PICK
       UTILS --> LOC
       UTILS --> MAG
       UTILS --> FOCAL
       FOCAL --> MT
       FOCAL --> STRESS
       LOC --> MAG
       LOC --> FOCAL
       UNC --> LOC
       UNC --> MAG
       CLI --> PIPE
