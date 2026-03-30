"""LLM injection configuration dataclass for maze_plr.py training loop integration.

Holds all parameters controlling when, how, and at what rate LLM-generated mazes
are injected into the PLR buffer during training. CLI flags on maze_plr.py populate
this config via from_config_dict().
"""

from dataclasses import dataclass


@dataclass
class LLMInjectionConfig:
    """Configuration for LLM-based maze injection into PLR training loop.

    All fields have sensible defaults. When using from_config_dict(), values are
    extracted from the CLI config dict produced by vars(parser.parse_args()).

    Phase 1 note: score_seeds_with_rollout defaults to False. Full policy rollout
    scoring (Tier 1 from CONTEXT.md) is deferred to Phase 2 when AgentEvaluator
    is wired in. Phase 1 uses max-priority insertion only.
    """

    # Core toggles
    enabled: bool = False              # --use_llm sets this True
    provider: str = ""                 # --llm_provider (required when enabled)
    model: str = ""                    # --llm_model
    config_path: str = ""              # --llm_config (path to llm/config.yaml)

    # Timing
    injection_interval: int = 3000     # --llm_inject_interval (eval steps between injections)
    inject_start_step: int = 5000      # --llm_inject_start_step (no injection before this training step)

    # LLM generation
    n_raw: int = 25                    # --llm_batch_size (mazes requested per injection)
    reference_maze_strategy: str = "hardest"  # --llm_ref_strategy
    n_reference_mazes: int = 5         # --llm_n_references

    # Diversity gate (Phase 2: enabled by default; locked decisions from CONTEXT.md)
    gate_enabled: bool = True               # --llm_gate
    difficulty_threshold: float = 0.6       # --llm_difficulty_threshold
    difficulty_metric: str = "sfl"         # --llm_difficulty_metric ("sfl" or "regret")
    difficulty_gate_mode: str = "fixed"     # --llm_difficulty_gate_mode
    #   "fixed": use difficulty_threshold as absolute floor
    #   "buffer_mean": threshold = mean score across entire buffer
    #   "reference_mean": threshold = mean score of N reference mazes
    #   "competitive": no difficulty gate, insert with actual SFL score (same as ACCEL)
    min_diversity: float = 0.02             # --llm_min_diversity
    diversity_gate_mode: str = "fixed"      # --llm_diversity_gate_mode
    #   "fixed": use min_diversity as absolute floor
    #   "buffer_median": threshold = median pairwise distance among N reference mazes
    #   "disabled": no diversity gate
    diversity_metric: str = "td_error_emd"  # --llm_diversity_metric
    max_diversity_retries: int = 2          # --llm_max_diversity_retries
    max_seed_retries: int = 3              # --llm_max_seed_retries (fresh LLM calls if gate rejects)
    n_rollouts_gate: int = 100              # --llm_n_rollouts (rollouts per candidate)

    # Independent mode: per-maze random refs, adaptive thresholds, no feedback
    independent: bool = False              # --llm_independent
    diversity_scale: float = 0.5           # --llm_diversity_scale
    difficulty_scale: float = 0.5          # --llm_difficulty_scale
    difficulty_floor: float = 0.05         # --llm_difficulty_floor
    max_independent_retries: int = 2       # --llm_max_independent_retries

    # Mutation amplification
    amplification_enabled: bool = True   # --llm_amplification
    mutations_per_seed: int = 30         # --llm_mutations_per_seed
    use_native_regret_filter: bool = True

    # Scoring strategy
    # NOTE: seed rollout scoring (Tier 1 from CONTEXT.md) is deferred to Phase 2
    # when AgentEvaluator is wired in. Phase 1 uses max-priority insertion only.
    score_seeds_with_rollout: bool = False    # Phase 2 sets True when AgentEvaluator exists
    score_mutations_with_rollout: bool = False
    mutations_solvability_check: bool = True

    # Buffer injection
    max_inject_per_event: int = 200    # --llm_max_inject_per_event

    # Tracking
    track_lineage: bool = True

    @classmethod
    def from_config_dict(cls, config: dict) -> "LLMInjectionConfig":
        """Construct LLMInjectionConfig from a CLI config dict.

        Maps CLI flag names (snake_case from argparse dest) to dataclass fields.
        Validates that --llm_provider is set when --use_llm is True.

        Args:
            config: dict from vars(parser.parse_args()) in maze_plr.py

        Returns:
            Populated LLMInjectionConfig instance

        Raises:
            ValueError: if use_llm=True and llm_provider is empty
        """
        instance = cls(
            enabled=config.get("use_llm", False),
            provider=config.get("llm_provider", ""),
            model=config.get("llm_model", ""),
            config_path=config.get("llm_config", ""),
            injection_interval=config.get("llm_inject_interval", 3000),
            inject_start_step=config.get("llm_inject_start_step", config.get("llm_warmup_steps", 5000)),
            n_raw=config.get("llm_batch_size", 25),
            reference_maze_strategy=config.get("llm_ref_strategy", "hardest"),
            n_reference_mazes=config.get("llm_n_references", 5),
            gate_enabled=config.get("llm_gate", True),
            difficulty_threshold=config.get("llm_difficulty_threshold", 0.6),
            difficulty_metric=config.get("llm_difficulty_metric", "sfl"),
            difficulty_gate_mode=config.get("llm_difficulty_gate_mode", "fixed"),
            min_diversity=config.get("llm_min_diversity", 0.02),
            diversity_gate_mode=config.get("llm_diversity_gate_mode", "fixed"),
            diversity_metric=config.get("llm_diversity_metric", "td_error_emd"),
            max_diversity_retries=config.get("llm_max_diversity_retries", 2),
            max_seed_retries=config.get("llm_max_seed_retries", 3),
            n_rollouts_gate=config.get("llm_n_rollouts", 100),
            independent=config.get("llm_independent", False),
            diversity_scale=config.get("llm_diversity_scale", 0.5),
            difficulty_scale=config.get("llm_difficulty_scale", 0.5),
            difficulty_floor=config.get("llm_difficulty_floor", 0.05),
            max_independent_retries=config.get("llm_max_independent_retries", 2),
            amplification_enabled=config.get("llm_amplification", True),
            mutations_per_seed=config.get("llm_mutations_per_seed", 30),
            max_inject_per_event=config.get("llm_max_inject_per_event", 200),
        )

        if instance.enabled and not instance.provider:
            raise ValueError(
                "--llm_provider is required when --use_llm is set"
            )

        return instance
