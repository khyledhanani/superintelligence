"""Build LLM prompts for maze generation with pluggable metric injection.

The prompt has three parts:
1. System prompt — maze format spec, constraints, what makes a good maze
2. Reference mazes — existing buffer levels with optional path overlays + metrics
3. Generation instruction — what to produce, informed by injected metrics

Metrics are fully pluggable: callers pass a list of MetricEntry objects that
get formatted into the prompt. This works for DTW metrics, regret, solve rate,
or any future metric.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class MetricEntry:
    """A single metric to inject into the prompt for a reference maze.

    Attributes:
        name: Human-readable metric name (e.g. "Position Trace DTW", "Regret")
        value: Scalar value or short string representation
        description: Optional one-line explanation of what this metric means
        higher_is: Optional hint — "better", "worse", "more diverse", etc.
    """
    name: str
    value: Any
    description: str = ""
    higher_is: str = ""
    metric_key: str = ""

    def format(self) -> str:
        parts = [f"  - {self.name}: {self._fmt_value()}"]
        if self.description:
            parts[0] += f"  ({self.description})"
        if self.higher_is:
            parts[0] += f"  [higher = {self.higher_is}]"
        return parts[0]

    def _fmt_value(self):
        if isinstance(self.value, float):
            return f"{self.value:.4f}"
        return str(self.value)


@dataclass
class ReferenceMaze:
    """A reference maze to include in the prompt.

    Attributes:
        grid: ASCII grid string (13x13, characters: #.>v<^G)
        label: Short label like "Maze A", "Maze 1"
        metrics: List of MetricEntry objects to display for this maze
        path_overlay: Optional ASCII grid with step numbers overlaid
        notes: Optional free-text notes about this maze
    """
    grid: str
    label: str = "Reference"
    metrics: List[MetricEntry] = field(default_factory=list)
    path_overlay: Optional[str] = None
    notes: str = ""


@dataclass
class PairwiseMetricEntry:
    """A pairwise metric between two reference mazes.

    Attributes:
        maze_a_label: Label of first maze
        maze_b_label: Label of second maze
        name: Metric name
        value: Scalar value
        description: What it means
    """
    maze_a_label: str
    maze_b_label: str
    name: str
    value: Any
    description: str = ""
    metric_key: str = ""

    def format(self) -> str:
        val = f"{self.value:.4f}" if isinstance(self.value, float) else str(self.value)
        line = f"  - {self.maze_a_label} vs {self.maze_b_label} — {self.name}: {val}"
        if self.description:
            line += f"  ({self.description})"
        return line


_SYSTEM_PROMPT_BASE = """You are a maze designer for a reinforcement learning environment.

MAZE FORMAT:
- Grid: exactly 13 rows x 13 columns
- Characters:
  # = wall (impassable)
  . = empty floor
  > = agent start (facing right)
  v = agent start (facing down)
  < = agent start (facing left)
  ^ = agent start (facing up)
  G = goal position
- Exactly ONE agent start and ONE goal position
- The outer border does NOT need to be all walls — open borders are fine
- The agent must be able to reach the goal (maze must be solvable)

DESIGN PRINCIPLES:
- Interesting mazes force the agent to navigate around obstacles
- Variety in path structure: corridors, open rooms, chokepoints, dead ends
- The agent start and goal should be separated by meaningful navigation
- Avoid trivial mazes (no walls) or impossible mazes (goal unreachable)"""

_OUTPUT_FORMAT_GRID_ONLY = """
OUTPUT FORMAT:
Return ONLY the 13x13 grid, one row per line, with no extra text before or after.
Do not wrap in code blocks or add any explanation.
Each row must be exactly 13 characters. There must be exactly 13 rows.
The output will be saved directly to a .txt file, so it must be a clean grid only.

Example of valid output format (this is deliberately trivial — generate something much more interesting):
#############
#>..........#
#...........#
#...........#
#...........#
#...........#
#.####.####.#
#...........#
#...........#
#...........#
#...........#
#..........G#
#############"""

_OUTPUT_FORMAT_WITH_REASONING = """
OUTPUT FORMAT:
First, write a brief reasoning section (3-5 sentences max), then output the grid.

REASONING (keep it short):
- What makes your maze different from the references
- Where you placed agent/goal and why
- How you verified solvability (trace the path mentally)

GRID:
Then output EXACTLY 13 rows of 13 characters each using only: # . > v < ^ G
One agent start (>v<^) and one goal (G). Must be solvable.

Example:

REASONING:
Agent bottom-left, goal top-right. Winding corridor with two dead ends.
Path: right along bottom, up through center gap, right to goal.

GRID:
#############
#>..........#
#...........#
#...........#
#...........#
#...........#
#.####.####.#
#...........#
#...........#
#...........#
#...........#
#..........G#
#############"""


def get_system_prompt(thinking_in_output: bool = False) -> str:
    """Build the system prompt with appropriate output format section."""
    if thinking_in_output:
        return _SYSTEM_PROMPT_BASE + _OUTPUT_FORMAT_WITH_REASONING
    return _SYSTEM_PROMPT_BASE + _OUTPUT_FORMAT_GRID_ONLY


# Keep backward compat
SYSTEM_PROMPT = _SYSTEM_PROMPT_BASE + _OUTPUT_FORMAT_GRID_ONLY


# ---------------------------------------------------------------------------
# Metric definitions — injected into prompts when a metric is active
# ---------------------------------------------------------------------------

METRIC_DEFINITIONS: Dict[str, tuple] = {
    "per_step_entropy": (
        "Per-Step Entropy",
        "Per-Step Entropy measures the RL agent's policy uncertainty at each timestep. "
        "High entropy (>0.5) means the agent is unsure which action to take — these are "
        "decision points or confusing junctions. Low entropy (<0.2) means the agent is "
        "confident about its path. To create higher entropy, design ambiguous branch points "
        "where multiple corridors look equally viable, or add deceptive openings near walls.",
    ),
    "per_step_regret": (
        "Per-Step Regret",
        "Per-Step Regret is (max_return - V(s_t)) at each timestep — how much the agent "
        "underestimates its potential from each state. This matches the ACCEL MaxMC formula "
        "but kept as a time series instead of averaged to a scalar. High regret at a step "
        "means the agent is confused about the maze's value at that point. To create high "
        "regret, add deceptive dead ends or long detours that mislead the agent's value "
        "estimate.",
    ),
    "scalar_regret": (
        "Scalar Regret (MaxMC)",
        "Scalar Regret (MaxMC) is the mean per-step regret across the full episode: "
        "mean_t[max_return - V(s_t)]. It summarizes overall maze difficulty for the agent. "
        "Higher regret = more learning potential (the agent's value estimates are far from "
        "reality). Regret near zero means the maze is too easy or already mastered. "
        "Mazes in the buffer typically have regret between 0.1 and 1.5.",
    ),
    "action_sequence": (
        "Action Sequence",
        "Action Sequence records the agent's discrete action at each timestep "
        "(0=up, 1=right, 2=down, 3=left, 4=stay). It is the behavioral fingerprint of "
        "the level — two mazes that produce the same action sequence are functionally "
        "identical from the agent's perspective, regardless of visual differences. "
        "To force different actions, change wall placement to block the current route "
        "and open alternative corridors.",
    ),
    "position_vector": (
        "Position Trace",
        "Position Trace records the agent's (x, y) grid coordinates at each timestep. "
        "Use this to link other metrics (regret, entropy, actions) to specific grid "
        "locations. Repeated positions mean the agent is stuck or looping. Compare "
        "position traces with the per-step regret or entropy vectors to identify exactly "
        "where on the grid the agent struggles.",
    ),
    "value_error": (
        "Value Error",
        "Value Error is the signed difference V(s_t) - G_t at each timestep, where G_t "
        "is the actual return from step t. Positive = agent overestimates (overconfident, "
        "walks into traps). Negative = agent underestimates (underconfident, doesn't "
        "recognize good positions). The sign reveals the nature of the error, not just "
        "its magnitude.",
    ),
    "solve_rate": (
        "Solve Rate",
        "Solve Rate is the fraction of rollouts where the agent reaches the goal. "
        "100% means the maze is too easy — the agent has mastered it. 0% means it's "
        "impossible or far too hard. The most useful mazes for training have solve rates "
        "between 30% and 70% — the agent can sometimes solve them but not always.",
    ),
    "learnability": (
        "SFL Learnability",
        "SFL Learnability is p × (1-p) where p is the agent's solve rate across many "
        "rollouts. Maximum learnability (0.25) occurs at p=0.5, meaning the agent solves "
        "the maze half the time — it's right at its learning frontier. Learnability near "
        "zero means the maze is either too easy (p≈1) or too hard (p≈0). Aim for mazes "
        "with learnability > 0.1 (solve rate between ~13% and ~87%).",
    ),
    "position_dtw": (
        "Position DTW",
        "Position DTW (Dynamic Time Warping) measures how similar the agent's spatial "
        "path is between two mazes. It compares the (x, y) trajectory on each maze after "
        "normalizing to start position (translation invariant). The distance is normalized "
        "by warping path length, so it is comparable across different episode lengths. "
        "Low DTW (<0.3) = agent walks nearly the same route on both mazes. High DTW (>0.5) "
        "= spatially distinct paths. To increase DTW vs a reference, rearrange walls so "
        "the agent must traverse completely different regions of the 13x13 grid.",
    ),
    "mode_transition": (
        "Experience Divergence",
        "Experience Divergence measures how differently the agent experiences two mazes "
        "using KL divergence between mode transition matrices. Each timestep is classified "
        "into one of 5 modes (confident_correct, confident_wrong, uncertain, recovering, "
        "degrading) based on value error and policy entropy. Higher divergence = the agent "
        "goes through fundamentally different learning processes on the two mazes.",
    ),
    "normalized_td_error": (
        "Normalized TD Error",
        "Normalized TD Error shows the fraction of total learning signal at each timestep: "
        "δ_t / Σ|δ|, where δ_t = r_t + γV(s_{t+1}) - V(s_t). A spike of 0.15 means 15% "
        "of all learning on this maze happens at that step (a critical decision point, trap, "
        "or surprise). Flat profile = learning is spread evenly. Spiky profile = a few key "
        "moments dominate what the agent learns. Use this to identify which steps matter most.",
    ),
    "td_error": (
        "Normalized TD Error EMD",
        "Normalized TD Error EMD is the Earth Mover's Distance between the normalized "
        "distributions of temporal difference errors (δ_t = r_t + γV(s_{t+1}) - V(s_t)) "
        "on two mazes. TD errors are divided by their total absolute sum before comparison, "
        "isolating the *shape* of the learning signal from its magnitude (since SFL "
        "learnability already captures how much learning happens). Higher EMD = more "
        "different learning signal shapes. This is the most task-agnostic diversity metric.",
    ),
    "cenie": (
        "CENIE Novelty",
        "CENIE Novelty measures how novel the agent's experience on a maze is compared to "
        "the entire training buffer. It fits a Gaussian Mixture Model on the agent's LSTM "
        "hidden states from past curriculum levels, then scores new levels by their negative "
        "log-likelihood under the GMM. Higher novelty = the agent enters unfamiliar states "
        "it hasn't encountered before. To increase novelty, design mazes that force the "
        "agent into unusual spatial patterns and decision sequences.",
    ),
}


def _collect_metric_keys(
    references: Optional[List[ReferenceMaze]] = None,
    pairwise_metrics: Optional[List[PairwiseMetricEntry]] = None,
    global_metrics: Optional[List[MetricEntry]] = None,
    target_metrics: Optional[List[MetricEntry]] = None,
    extra_keys: Optional[List[str]] = None,
) -> List[str]:
    """Collect unique metric_keys from all metric objects, preserving insertion order."""
    seen = {}
    for source in [global_metrics, target_metrics]:
        if source:
            for m in source:
                if m.metric_key and m.metric_key not in seen:
                    seen[m.metric_key] = True
    if references:
        for ref in references:
            for m in ref.metrics:
                if m.metric_key and m.metric_key not in seen:
                    seen[m.metric_key] = True
    if pairwise_metrics:
        for pm in pairwise_metrics:
            if pm.metric_key and pm.metric_key not in seen:
                seen[pm.metric_key] = True
    if extra_keys:
        for k in extra_keys:
            if k and k not in seen:
                seen[k] = True
    return list(seen.keys())


def _render_definitions_section(keys: List[str]) -> str:
    """Render a METRIC DEFINITIONS section for the given keys.

    Returns empty string if no keys have definitions.
    """
    definitions = []
    for key in keys:
        if key in METRIC_DEFINITIONS:
            title, body = METRIC_DEFINITIONS[key]
            definitions.append(f"{title}: {body}")
    if not definitions:
        return ""
    header = (
        "=== METRIC DEFINITIONS ===\n"
        "The following metrics describe how the RL agent behaves on each maze. "
        "Use these to understand the data below and to guide your maze design.\n"
    )
    return header + "\n\n".join(definitions)


def overlay_path_on_grid(grid_str: str, positions: np.ndarray) -> str:
    """Overlay agent path step numbers onto an ASCII maze grid.

    Args:
        grid_str: 13x13 ASCII maze string
        positions: (T, 2) array of (x, y) positions

    Returns:
        ASCII grid with step numbers on visited cells.
        Step numbers use single chars: 0-9, then a-z, then A-Z, then *.
    """
    rows = grid_str.strip().split('\n')
    grid = [list(row) for row in rows]

    step_chars = (
        [str(i) for i in range(10)]
        + [chr(ord('a') + i) for i in range(26)]
        + [chr(ord('A') + i) for i in range(26)]
    )

    for step, (x, y) in enumerate(positions):
        x, y = int(x), int(y)
        if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
            char = step_chars[step] if step < len(step_chars) else '*'
            grid[y][x] = char

    return '\n'.join(''.join(row) for row in grid)


def build_generation_prompt(
    references: List[ReferenceMaze],
    pairwise_metrics: Optional[List[PairwiseMetricEntry]] = None,
    global_metrics: Optional[List[MetricEntry]] = None,
    instruction: str = "",
    target_metrics: Optional[List[MetricEntry]] = None,
) -> str:
    """Build the user prompt for maze generation.

    Args:
        references: Reference mazes with their metrics
        pairwise_metrics: Optional pairwise metrics between reference mazes
        global_metrics: Optional buffer-wide summary metrics
        instruction: Custom instruction appended to the prompt.
            If empty, a default instruction is used.
        target_metrics: Optional target metric values to aim for

    Returns:
        User prompt string (system prompt is returned separately)
    """
    sections = []

    # Section 0: Metric definitions (only for metrics actually used)
    keys = _collect_metric_keys(references, pairwise_metrics, global_metrics, target_metrics)
    defs = _render_definitions_section(keys)
    if defs:
        sections.append(defs)
        sections.append("")

    # Section 1: Reference mazes
    if references:
        sections.append("=== REFERENCE MAZES FROM THE REPLAY BUFFER ===")
        sections.append("These are mazes the RL agent is currently training on.\n")

        for ref in references:
            sections.append(f"--- {ref.label} ---")
            sections.append(ref.grid)

            if ref.path_overlay:
                sections.append(f"\nAgent path overlay for {ref.label}:")
                sections.append(ref.path_overlay)

            if ref.metrics:
                sections.append(f"\nMetrics for {ref.label}:")
                for m in ref.metrics:
                    sections.append(m.format())

            if ref.notes:
                sections.append(f"\nNote: {ref.notes}")

            sections.append("")  # blank line

    # Section 2: Pairwise metrics
    if pairwise_metrics:
        sections.append("=== PAIRWISE DIVERSITY METRICS ===")
        for pm in pairwise_metrics:
            sections.append(pm.format())
        sections.append("")

    # Section 3: Global buffer metrics
    if global_metrics:
        sections.append("=== BUFFER-WIDE METRICS ===")
        for gm in global_metrics:
            sections.append(gm.format())
        sections.append("")

    # Section 4: Target metrics
    if target_metrics:
        sections.append("=== TARGET METRICS FOR NEW MAZE ===")
        sections.append("Generate a maze that aims for these metric values:")
        for tm in target_metrics:
            sections.append(tm.format())
        sections.append("")

    # Section 5: Generation instruction
    if instruction:
        sections.append("=== INSTRUCTION ===")
        sections.append(instruction)
    else:
        sections.append("=== INSTRUCTION ===")
        sections.append(
            "Generate a NEW 13x13 maze that is DIFFERENT from the reference mazes above. "
            "The new maze should provide a distinct navigation challenge — different path "
            "structure, different obstacle layout, different spatial regions explored. "
            "Make sure it is solvable (agent can reach the goal)."
        )

    return '\n'.join(sections)


def build_feedback_prompt(
    candidate_grid: str,
    error_message: str,
    original_instruction: str = "",
) -> str:
    """Build a follow-up prompt when a candidate maze fails validation.

    Args:
        candidate_grid: The candidate maze that failed
        error_message: Specific error description
        original_instruction: The original generation instruction (for context)

    Returns:
        Follow-up user prompt
    """
    sections = [
        "Your previous maze had an issue:\n",
        candidate_grid,
        f"\nPROBLEM: {error_message}\n",
        "Please fix this and generate a corrected 13x13 maze. "
        "Return ONLY the grid, one row per line.",
    ]
    return '\n'.join(sections)


def build_diversity_feedback_prompt(
    candidate_grid: str,
    candidate_overlay: Optional[str],
    similarity_issues: List[str],
    analysis_sections: Optional[List] = None,
    reference_overlays: Optional[Dict[str, str]] = None,
    metric_keys: Optional[List[str]] = None,
) -> str:
    """Build feedback when a candidate passes validation but fails the diversity gate.

    Args:
        candidate_grid: The valid but too-similar candidate
        candidate_overlay: Optional path overlay showing agent behavior
        similarity_issues: List of specific similarity problems
        analysis_sections: Optional list of AnalysisSection objects from pluggable
            analyzers. Each section is rendered with its own header between the
            similarity issues and the final instruction. This is the extension
            point for metric-specific spatial/causal reasoning.
        reference_overlays: Optional dict of {label: path_overlay} for reference
            mazes that the candidate is most similar to, so the LLM can visually
            compare agent paths.
        metric_keys: List of metric keys used in the gate evaluation. Definitions
            for these metrics are injected at the top of the feedback prompt.

    Returns:
        Follow-up user prompt
    """
    sections = [
        "Your maze is valid but too similar to existing buffer mazes:\n",
    ]

    # Inject metric definitions so the LLM understands the feedback
    if metric_keys:
        defs = _render_definitions_section(metric_keys)
        if defs:
            sections.append(defs)
            sections.append("")

    sections.append(candidate_grid)

    if candidate_overlay:
        sections.append("\nAgent path on your maze:")
        sections.append(candidate_overlay)

    # Show reference maze path overlays for visual comparison
    if reference_overlays:
        for label, overlay in reference_overlays.items():
            sections.append(f"\nAgent path on {label} (for comparison):")
            sections.append(overlay)

    sections.append("\nSIMILARITY ISSUES:")
    for issue in similarity_issues:
        sections.append(f"  - {issue}")

    # Inject pluggable analysis sections (spatial reasoning, value analysis, etc.)
    if analysis_sections:
        for section in analysis_sections:
            sections.append(f"\n=== {section.title} ===")
            sections.append(section.body)

    if analysis_sections:
        sections.append(
            "\nGenerate a MORE DIFFERENT maze. Use the analysis above to guide "
            "your wall placement — block the identified overlap regions and open "
            "paths through unused regions. Return ONLY the 13x13 grid."
        )
    else:
        sections.append(
            "\nGenerate a MORE DIFFERENT maze. Change the wall structure to force "
            "the agent into a completely different navigation path. "
            "Return ONLY the 13x13 grid."
        )

    return '\n'.join(sections)
