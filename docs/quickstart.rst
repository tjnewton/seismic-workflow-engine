Quick Start
===========

This guide walks through a basic SWE workflow: downloading data, picking
phases, and locating events.

Download waveforms
------------------

.. code-block:: python

    from obspy import UTCDateTime
    from swe.io.fdsn import get_waveforms

    paths = get_waveforms(
        network="CI",
        stations=["WBS", "WRC2"],
        channels=["BHZ"],
        start_time=UTCDateTime("2019-07-04T17:30:00"),
        end_time=UTCDateTime("2019-07-04T18:30:00"),
        output_dir="data/waveforms",
        client="SCEDC",
    )

Phase picking
-------------

.. code-block:: python

    from swe.picking import find_picking_windows

    filepaths = sorted(Path("data/waveforms").glob("*.ms"))
    result = find_picking_windows(filepaths, method="sta_lta")

Event location
--------------

.. code-block:: python

    from swe.location.nonlinloc import catalog_to_obs, run_nonlinloc

    catalog_to_obs(catalog, output_path="nll_picks.obs")
    run_nonlinloc("nll_control.txt")

CLI usage
---------

SWE also provides a command-line interface::

    swe --help
    swe download --network CI --station WBS --channel BHZ \
        --start 2019-07-04T17:30:00 --end 2019-07-04T18:30:00
    swe run examples/pipeline_config.yml

Uncertainty propagation
-----------------------

When pick-time uncertainties are available, propagate them through
location via Monte Carlo relocation:

.. code-block:: python

    from swe.location import locate_by_travel_time_with_uncertainty

    result = locate_by_travel_time_with_uncertainty(
        picks=picks,
        velocity_model=model,
        pick_time_uncertainties=uncertainties,
    )
    print(result.uncertainty.horiz_uncertainty_m)

See :doc:`cli` for the full CLI reference.
