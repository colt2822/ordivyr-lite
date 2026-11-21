"""ORDIVYR Lite - a terminal-first, standalone AI execution engine.

Lite owns the execution loop: context, tools, model routing, validation,
recovery and transparent usage accounting.  It has no runtime dependency on
the ORDIVYR Board, its web server, database or scheduling layer.

The public entry points are:
  * ``ordivyr_lite.cli:main`` - the ``ordivyr-lite`` command line
  * ``ordivyr_lite.loop.run`` - one programmatic execution
  * ``ordivyr_lite.job``     - the versioned JSON job/result contract
"""

__version__ = "0.2.0"
JOB_PROTOCOL = "ordivyr-lite/job"
RESULT_PROTOCOL = "ordivyr-lite/result"
PROTOCOL_VERSION = 1

__all__ = ["__version__", "JOB_PROTOCOL", "RESULT_PROTOCOL", "PROTOCOL_VERSION"]
