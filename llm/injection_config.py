"""LLM injection configuration dataclass for maze_plr.py training loop integration.

Holds parameters controlling when, how, and at what rate LLM-generated mazes
are injected into the PLR buffer during training.

Shared LLM generation parameters (provider, model, reference strategy, gate
thresholds, prompt metrics) live in config.yaml — this dataclass only holds
injection-specific parameters that have no equivalent in config.yaml.

CLI flags on maze_plr.py can override any config.yaml value.
"""

import os
from dataclasses import dataclass


@dataclass
class LLMInjectionConfig:
    """Configuration for LLM-based maze injection into PLR training loop.

    Shared parameters (reference strategy, gate thresholds, etc.) are loaded
    from config.yaml via from_config_dict(). Injection-specific parameters
    (timing, batch size, mutation, buffer targets) are set here.
    """

    # Core toggles
    enabled: bool = True              # --use_llm
    config_path: str = ""              # --llm_config (path to llm/config.yaml)

    # --- Injection-specific (no config.yaml equivalent) ---

    # Timing
    injection_interval: int = 5000     # --llm_inject_interval (eval steps between injections)
    inject_start_step: int = 5000      # --llm_inject_start_step

    # Batch sizing
    n_raw: int = 10                    # --llm_batch_size (mazes per injection round)
    target_buffer_pct: float = 0.05     # --llm_target_buffer_pct (0=disabled, 0.05=5%)

    # Mutation amplification
    amplification_enabled: bool = True   # --llm_amplification
    mutations_per_seed: int = 30         # --llm_mutations_per_seed
    mutations_solvability_check: bool = True

    # --- Shared with config.yaml (CLI overrides YAML) ---
    # These are populated from config.yaml with CLI flag overrides.

    buffer_state: str = ""             # "stale" or "fresh" (default from config.yaml)
    provider: str = ""                 # --llm_provider
    model: str = ""                    # --llm_model
    reference_maze_strategy: str = ""  # --llm_ref_strategy (default from config.yaml)
    n_reference_mazes: int = 0         # --llm_n_references (default from config.yaml)
    hybrid_difficulty_percentile: float = 0.0  # --llm_hybrid_difficulty_percentile

    # Gate
    gate_enabled: bool = True               # --llm_gate
    difficulty_threshold: float = 0.0       # --llm_difficulty_threshold
    difficulty_gate_mode: str = ""          # --llm_difficulty_gate_mode
    min_diversity: float = 0.0              # --llm_min_diversity
    diversity_gate_mode: str = ""           # --llm_diversity_gate_mode
    diversity_metric: str = ""              # --llm_diversity_metric
    max_diversity_retries: int = 0          # --llm_max_diversity_retries
    n_rollouts_gate: int = 0               # --llm_n_rollouts

    @classmethod
    def from_config_dict(cls, config: dict) -> "LLMInjectionConfig":
        """Construct LLMInjectionConfig from CLI config dict + config.yaml.

        Loads config.yaml (via --llm_config path) for shared defaults,
        then applies CLI flag overrides.

        Args:
            config: dict from vars(parser.parse_args()) in maze_plr.py

        Returns:
            Populated LLMInjectionConfig instance

        Raises:
            ValueError: if use_llm=True and llm_provider is empty
        """
        # Load config.yaml defaults
        yaml_cfg = {}
        config_path = config.get("llm_config", "")
        if config_path and os.path.exists(config_path):
            import yaml
            with open(config_path) as f:
                yaml_cfg = yaml.safe_load(f) or {}

        gate_cfg = yaml_cfg.get("gate", {})

        # Helper: CLI value wins if present, else YAML value, else hardcoded default.
        # Uses None as sentinel — argparse defaults should be None for shared fields
        # so we can distinguish "not set" from "set to 0".
        def _resolve(cli_key, yaml_key, default, yaml_dict=yaml_cfg):
            cli_val = config.get(cli_key)
            if cli_val is not None:
                return cli_val
            yaml_val = yaml_dict.get(yaml_key)
            if yaml_val is not None:
                return yaml_val
            return default

        instance = cls(
            # Core
            enabled=config.get("use_llm", False),
            config_path=config_path,

            # Injection-specific (CLI only)
            injection_interval=config.get("llm_inject_interval", 5000),
            inject_start_step=config.get("llm_inject_start_step", config.get("llm_warmup_steps", 5000)),
            n_raw=config.get("llm_batch_size", 25),
            target_buffer_pct=config.get("llm_target_buffer_pct", 0.0),
            amplification_enabled=config.get("llm_amplification", True),
            mutations_per_seed=config.get("llm_mutations_per_seed", 30),
            buffer_state=_resolve("llm_buffer_state", "buffer_state", "stale"),

            # Shared: CLI flag → config.yaml → hardcoded default
            provider=_resolve("llm_provider", "provider", ""),
            model=_resolve("llm_model", "model", ""),
            reference_maze_strategy=_resolve("llm_ref_strategy", "strategy", "hardest"),
            n_reference_mazes=_resolve("llm_n_references", "num_refs", 5),
            hybrid_difficulty_percentile=_resolve("llm_hybrid_difficulty_percentile", "hybrid_difficulty_percentile", 50.0),

            # Gate: CLI flag → config.yaml gate: sub-dict → hardcoded default
            gate_enabled=_resolve("llm_gate", "enabled", True, gate_cfg),
            difficulty_threshold=_resolve("llm_difficulty_threshold", "difficulty_threshold", 0.1, gate_cfg),
            difficulty_gate_mode=_resolve("llm_difficulty_gate_mode", "difficulty_gate_mode", "buffer_mean", gate_cfg),
            min_diversity=_resolve("llm_min_diversity", "min_diversity", 0.4, gate_cfg),
            diversity_gate_mode=_resolve("llm_diversity_gate_mode", "diversity_gate_mode", "fixed", gate_cfg),
            diversity_metric=_resolve("llm_diversity_metric", "diversity_metric", "embedding_l2", gate_cfg),
            max_diversity_retries=_resolve("llm_max_diversity_retries", "max_diversity_retries", 3, gate_cfg),
            n_rollouts_gate=_resolve("llm_n_rollouts", "n_rollouts", 50, gate_cfg),
        )

        if instance.enabled and not instance.provider:
            raise ValueError(
                "--llm_provider is required when --use_llm is set"
            )

        return instance
