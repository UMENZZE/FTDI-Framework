# FTDI: Fault-Tolerant Diagnosis and Iterative Repair Framework
# Core components for multi-agent system resilience
#
# Modules:
#   auditor          - Heuristic code auditor (non-LLM scoring)
#   repair_agent     - T0 regex + T2 LLM repair agent
#   repair_strategy  - Error type -> repair strategy mapping
#   tiered_repair    - Tiered repair dispatch (repair_if_needed)
#   hook             - Environment monkey-patch orchestration
