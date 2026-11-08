"""jmon: testbed for budgeted joint monitoring allocation in multi-agent LLM systems.

Design rules (see research/research_proposal.md):
- Agent communication is simulated once per (config, seed) and logged. Monitoring is
  applied to the log afterwards, so changing the monitoring assignment can never change
  what the agents said.
- A monitor sees only the messages on the edges assigned to it; the view is built by
  ``monitors.views.build_view`` and tests assert the restriction.
- Every monitor call is charged to a ``BudgetLedger``.
"""
__version__ = "0.0.1"
