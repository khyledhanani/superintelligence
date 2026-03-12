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

    def format(self) -> str:
        val = f"{self.value:.4f}" if isinstance(self.value, float) else str(self.value)
        line = f"  - {self.maze_a_label} vs {self.maze_b_label} — {self.name}: {val}"
        if self.description:
            line += f"  ({self.description})"
        return line


SYSTEM_PROMPT = """You are a maze designer for a reinforcement learning environment.

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
- Avoid trivial mazes (no walls) or impossible mazes (goal unreachable)

OUTPUT FORMAT:
Return ONLY the 13x13 grid, one row per line, with no extra text before or after.
Do not wrap in code blocks or add any explanation.
Each row must be exactly 13 characters. There must be exactly 13 rows.
The output will be saved directly to a .txt file, so it must be a clean grid only.

Example of a valid output (do NOT copy this maze, create a new one):
#############
#>..........#
#.#########.#
#.#.......#.#
#.#.#####.#.#
#.#.#...#.#.#
#.#.#.#.#.#.#
#.#...#...#.#
#.#########.#
#...........#
###########.#
#G..........#
#############"""


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

    Returns:
        Follow-up user prompt
    """
    sections = [
        "Your maze is valid but too similar to existing buffer mazes:\n",
        candidate_grid,
    ]

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
