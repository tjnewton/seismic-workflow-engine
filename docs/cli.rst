CLI Reference
=============

SWE provides a command-line interface via the ``swe`` command.

Top-level options
-----------------

.. code-block:: text

    swe [OPTIONS] COMMAND [ARGS]...

    Options:
      --version      Show the version and exit.
      -v, --verbose  Increase verbosity (-v for INFO, -vv for DEBUG).
      -h, --help     Show this message and exit.

Commands
--------

``swe run``
~~~~~~~~~~~

Execute a YAML pipeline configuration::

    swe run CONFIG_FILE [OPTIONS]

Options:

- ``--skip STEP_NAME``: skip one or more named steps (repeatable)
- ``--propagate-uncertainty``: propagate pick-time uncertainties through location and magnitude steps
- ``--no-cache``: disable output caching for this run
- ``--parallel``: run independent steps concurrently
- ``--resume RUN_ID``: resume from a prior run (or ``latest``)
- ``--run-dir PATH``: base directory for run state (default: ``.swe_runs``)
- ``--show-provenance RUN_ID``: display provenance records for a run and exit without executing (or ``latest``)
- ``--dry-run``: validate pipeline YAML without executing (checks imports, step references, cycles)

See :doc:`quickstart` for pipeline YAML syntax.

``swe pipeline template``
~~~~~~~~~~~~~~~~~~~~~~~~~

Scaffold a starter YAML configuration for a named workflow::

    swe pipeline template WORKFLOW [-o OUTPUT]

Available workflows: ``full``, ``detection``, ``stress``, ``spectral``,
``magnitude``, ``uncertainty``, ``roughness``.

``swe pipeline list``
~~~~~~~~~~~~~~~~~~~~~

List all available pipeline templates::

    swe pipeline list

``swe download``
~~~~~~~~~~~~~~~~

Download waveforms and station metadata from FDSN::

    swe download --network CI --station WBS --channel BHZ \
        --start 2019-07-04T17:30:00 --end 2019-07-04T18:30:00 \
        --output-dir data/waveforms

``swe noise``
~~~~~~~~~~~~~

Classify ambient noise in continuous waveforms::

    swe noise --input-dir data/waveforms --start 2019-07-04T17:30:00 \
        --end 2019-07-04T18:30:00 --window-length 240

``swe pick``
~~~~~~~~~~~~

Run automated phase picking::

    swe pick --input-dir data/waveforms --method stalta --output picks.csv

``swe associate``
~~~~~~~~~~~~~~~~~

Associate phase picks into events::

    swe associate --picks picks.csv --stations stations.csv --output catalog.xml

``swe detect``
~~~~~~~~~~~~~~

Run matched-filter detection::

    swe detect --marker-file picks.mrkr --template-dir templates/ \
        --data-dir data/waveforms --start 2019-07-04 --end 2019-07-05

``swe locate``
~~~~~~~~~~~~~~

Locate events using NonLinLoc::

    swe locate --obs-file nll_picks.obs --control-file nll_control.txt

``swe plot``
~~~~~~~~~~~~

Generate waveform, spectrogram, or spectrum plots::

    swe plot --input data.ms --plot-type waveform --output plot.png

Pipeline YAML Format
--------------------

A pipeline configuration is a YAML file with a ``steps`` list.
Each step has ``name``, ``function``, and ``params``:

.. code-block:: yaml

    steps:
      - name: download_waveforms
        function: swe.io.fdsn.get_waveforms
        params:
          network: "CI"
          stations: ["WBS"]
          channels: ["BHZ"]
          start_time: "2019-07-04T17:30:00"
          end_time: "2019-07-04T18:30:00"
          output_dir: "data/waveforms"

Parameters can reference outputs from prior steps using ``${step_name.key}``
syntax.
