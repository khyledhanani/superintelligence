"""LLM-based maze generator using Ollama cloud API.

Generates 13x13 maze levels by prompting an LLM with reference mazes and
pluggable metrics, then validates and parses the output.

Uses Ollama's native /api/chat endpoint (not OpenAI-compatible).
API key loaded from OLLAMA_API_KEY env var or .env file.

Usage:
    generator = MazeGenerator()
    result = generator.generate(references=[...], metrics=[...])
    if result.success:
        level = result.level  # Level object ready for the replay buffer
"""

import os
import re
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable

import requests
import numpy as np

from llm.prompt_builder import (
    SYSTEM_PROMPT,
    ReferenceMaze,
    MetricEntry,
    PairwiseMetricEntry,
    build_generation_prompt,
    build_feedback_prompt,
    build_diversity_feedback_prompt,
)

logger = logging.getLogger(__name__)


def _load_api_key(env_var: str = "OLLAMA_API_KEY") -> str:
    """Load API key from environment variable or .env file."""
    key = os.environ.get(env_var, "")
    if key:
        return key
    # Try .env file in project root
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{env_var}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


@dataclass
class GenerationConfig:
    """Configuration for the LLM maze generator.

    All values are loaded from llm/config.yaml and/or CLI flags.
    The dataclass holds no provider-specific defaults — config.yaml is
    the single source of truth.
    """
    provider: str = ""
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    temperature: float = 0.0
    max_retries: int = 0
    timeout: int = 0
    min_walls: int = 0
    min_path_distance: int = 0
    validate_solvable: bool = True

    # Provider defaults (used only as fallback when no config file is loaded)
    _PROVIDER_DEFAULTS = {
        "ollama": {
            "base_url": "https://ollama.com",
            "api_key_env": "OLLAMA_API_KEY",
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        },
    }

    def __post_init__(self):
        if not self.provider:
            self.provider = "ollama"
        defaults = self._PROVIDER_DEFAULTS.get(self.provider, {})
        if not self.base_url:
            self.base_url = defaults.get("base_url", "")
        if not self.api_key:
            env_var = defaults.get("api_key_env", "")
            if env_var:
                self.api_key = _load_api_key(env_var)
        if not self.api_key:
            logger.warning(
                f"No API key found for provider '{self.provider}'. "
                f"Set {defaults.get('api_key_env', 'API_KEY')} via environment variable or .env file."
            )


@dataclass
class GenerationResult:
    """Result of a maze generation attempt.

    Attributes:
        success: Whether a valid maze was produced
        grid: The ASCII grid string (if successful)
        level: The parsed Level object (if successful and parsing worked)
        attempts: Number of attempts made
        errors: List of error messages from failed attempts
        raw_responses: Raw LLM response strings
        latency_ms: Total time spent in ms
        gate_metrics: Diversity metrics from decision gate (if evaluated)
        diversity_attempts: Number of diversity gate attempts
        diversity_issues: Unresolved diversity issues (if gate failed)
    """
    success: bool = False
    grid: Optional[str] = None
    level: Any = None  # Level object (avoid circular import at module level)
    attempts: int = 0
    errors: List[str] = field(default_factory=list)
    raw_responses: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    gate_metrics: Optional[Dict] = None
    gate_pair_metrics: Optional[List] = None  # List[PairGateMetrics] with local_costs vectors
    diversity_attempts: int = 0
    diversity_issues: List[str] = field(default_factory=list)
    feedback_prompts: List[str] = field(default_factory=list)  # diversity feedback prompts sent to LLM


class MazeGenerator:
    """Generates maze levels via LLM with configurable metric injection."""

    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()

    def generate(
        self,
        references: Optional[List[ReferenceMaze]] = None,
        pairwise_metrics: Optional[List[PairwiseMetricEntry]] = None,
        global_metrics: Optional[List[MetricEntry]] = None,
        instruction: str = "",
        target_metrics: Optional[List[MetricEntry]] = None,
        solvability_checker: Optional[Callable] = None,
    ) -> GenerationResult:
        """Generate a new maze level via LLM.

        Args:
            references: Reference mazes with metrics for context
            pairwise_metrics: Pairwise diversity metrics between references
            global_metrics: Buffer-wide summary metrics
            instruction: Custom generation instruction
            target_metrics: Target metric values for the new maze
            solvability_checker: Optional callable(Level) -> bool for solvability.
                If None and config.validate_solvable is True, uses BFS flood fill.

        Returns:
            GenerationResult with the generated maze (or error details)
        """
        start = time.time()
        result = GenerationResult()

        user_prompt = build_generation_prompt(
            references=references or [],
            pairwise_metrics=pairwise_metrics,
            global_metrics=global_metrics,
            instruction=instruction,
            target_metrics=target_metrics,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(1, self.config.max_retries + 1):
            result.attempts = attempt

            # Call LLM
            raw_response = self._call_llm(messages)
            if raw_response is None:
                result.errors.append(f"Attempt {attempt}: LLM API call failed")
                continue
            result.raw_responses.append(raw_response)

            # Parse grid from response
            grid, parse_error = self._parse_grid(raw_response)
            if grid is None:
                error_msg = f"Attempt {attempt}: {parse_error}"
                result.errors.append(error_msg)
                logger.info(error_msg)
                # Add feedback for retry
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({
                    "role": "user",
                    "content": build_feedback_prompt(raw_response, parse_error),
                })
                continue

            # Validate grid format
            valid, format_error = self._validate_format(grid)
            if not valid:
                error_msg = f"Attempt {attempt}: {format_error}"
                result.errors.append(error_msg)
                logger.info(error_msg)
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({
                    "role": "user",
                    "content": build_feedback_prompt(grid, format_error),
                })
                continue

            # Parse to Level object
            level, level_error = self._parse_level(grid)
            if level is None:
                error_msg = f"Attempt {attempt}: {level_error}"
                result.errors.append(error_msg)
                logger.info(error_msg)
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({
                    "role": "user",
                    "content": build_feedback_prompt(grid, level_error),
                })
                continue

            # Validate complexity
            complexity_ok, complexity_error = self._validate_complexity(grid, level)
            if not complexity_ok:
                error_msg = f"Attempt {attempt}: {complexity_error}"
                result.errors.append(error_msg)
                logger.info(error_msg)
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({
                    "role": "user",
                    "content": build_feedback_prompt(grid, complexity_error),
                })
                continue

            # Validate solvability
            if self.config.validate_solvable:
                if solvability_checker is not None:
                    solvable = solvability_checker(level)
                else:
                    solvable = self._bfs_solvable(grid, level)

                if not solvable:
                    error_msg = f"Attempt {attempt}: Maze is not solvable — agent cannot reach the goal"
                    result.errors.append(error_msg)
                    logger.info(error_msg)
                    messages.append({"role": "assistant", "content": raw_response})
                    messages.append({
                        "role": "user",
                        "content": build_feedback_prompt(
                            grid,
                            "The maze is NOT solvable — the agent cannot reach the goal. "
                            "Make sure there is a clear path of '.' cells from the agent to 'G'.",
                        ),
                    })
                    continue

            # Success!
            result.success = True
            result.grid = grid
            result.level = level
            logger.info(f"Generated valid maze on attempt {attempt}")
            break

        result.latency_ms = (time.time() - start) * 1000
        return result

    def _call_llm(self, messages: List[Dict]) -> Optional[str]:
        """Call the LLM API. Dispatches to Ollama or OpenAI-compatible format."""
        if self.config.provider == "openrouter":
            return self._call_openai_compatible(messages)
        return self._call_ollama(messages)

    def _call_ollama(self, messages: List[Dict]) -> Optional[str]:
        """Call the Ollama cloud API (/api/chat endpoint)."""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
            },
        }

        url = f"{self.config.base_url}/api/chat"
        try:
            logger.debug(f"Calling {url} with model={self.config.model}")
            resp = requests.post(
                url, json=payload, headers=headers, timeout=self.config.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            # Ollama native format: response is in data["message"]["content"]
            # Some thinking models put output in "thinking" field with empty "content"
            content = data["message"].get("content", "")
            if not content.strip():
                thinking = data["message"].get("thinking", "")
                if thinking.strip():
                    logger.info("Content was empty, using thinking field as fallback")
                    content = thinking
            return content
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text[:500]}")
            return None
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected API response format: {e}")
            logger.error(f"Response data: {data}")
            return None

    def _call_openai_compatible(self, messages: List[Dict]) -> Optional[str]:
        """Call an OpenAI-compatible API (OpenRouter, etc.)."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }

        url = f"{self.config.base_url}/chat/completions"
        try:
            logger.debug(f"Calling {url} with model={self.config.model}")
            resp = requests.post(
                url, json=payload, headers=headers, timeout=self.config.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text[:500]}")
            return None
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected API response format: {e}")
            logger.error(f"Response data: {data}")
            return None

    def _parse_grid(self, raw_response: str) -> tuple:
        """Extract a 13x13 grid from the LLM response.

        Handles common LLM quirks: code blocks, extra text, whitespace.

        Returns:
            (grid_string, None) on success, (None, error_message) on failure
        """
        text = raw_response.strip()

        # Strip markdown code blocks if present
        code_block = re.search(r'```(?:\w*\n)?(.*?)```', text, re.DOTALL)
        if code_block:
            text = code_block.group(1).strip()

        # Find lines that look like maze rows (only valid chars, ~13 chars)
        valid_chars = set('#.>v<^G')
        maze_lines = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Check if line contains only valid maze characters
            if all(c in valid_chars for c in line) and 10 <= len(line) <= 16:
                maze_lines.append(line)

        if len(maze_lines) < 13:
            return None, (
                f"Found only {len(maze_lines)} valid maze rows (need exactly 13). "
                f"Each row must be exactly 13 characters using only: # . > v < ^ G"
            )

        # Take the first 13 valid rows
        maze_lines = maze_lines[:13]

        # Check all rows are same length
        lengths = [len(line) for line in maze_lines]
        if len(set(lengths)) > 1:
            bad_rows = [f"row {i+1} has {l} chars" for i, l in enumerate(lengths) if l != 13]
            return None, f"Rows have inconsistent lengths: {', '.join(bad_rows)}. All rows must be exactly 13 characters."

        if lengths[0] != 13:
            return None, f"Rows are {lengths[0]} characters wide, need exactly 13."

        grid = '\n'.join(maze_lines)
        return grid, None

    def _validate_format(self, grid: str) -> tuple:
        """Validate maze grid format constraints.

        Returns:
            (True, None) on success, (False, error_message) on failure
        """
        rows = grid.split('\n')

        # Count agents and goals
        agent_count = sum(row.count(c) for row in rows for c in '>v<^')
        goal_count = sum(row.count('G') for row in rows)

        if agent_count == 0:
            return False, "No agent start position found. Use one of: > v < ^ for the agent."
        if agent_count > 1:
            return False, f"Found {agent_count} agent positions, need exactly 1."
        if goal_count == 0:
            return False, "No goal position 'G' found."
        if goal_count > 1:
            return False, f"Found {goal_count} goals, need exactly 1."

        return True, None

    def _parse_level(self, grid: str) -> tuple:
        """Parse grid string into a Level object.

        Returns:
            (Level, None) on success, (None, error_message) on failure
        """
        try:
            from jaxued.environments.maze import Level
            level = Level.from_str(grid)
            return level, None
        except AssertionError as e:
            return None, f"Level parsing failed: {e}"
        except Exception as e:
            return None, f"Unexpected error parsing level: {e}"

    def _validate_complexity(self, grid: str, level) -> tuple:
        """Check that the maze meets minimum complexity requirements.

        Returns:
            (True, None) on success, (False, error_message) on failure
        """
        # Count walls
        wall_count = grid.count('#')
        if wall_count < self.config.min_walls:
            return False, (
                f"Maze has only {wall_count} walls, need at least {self.config.min_walls}. "
                "Add more walls to create interesting navigation challenges."
            )

        # Check Manhattan distance between agent and goal
        ax, ay = int(level.agent_pos[0]), int(level.agent_pos[1])
        gx, gy = int(level.goal_pos[0]), int(level.goal_pos[1])
        manhattan = abs(ax - gx) + abs(ay - gy)
        if manhattan < self.config.min_path_distance:
            return False, (
                f"Agent and goal are only {manhattan} cells apart (Manhattan distance). "
                f"Need at least {self.config.min_path_distance}. "
                "Place them further apart for a more meaningful challenge."
            )

        return True, None

    def _bfs_solvable(self, grid: str, level) -> bool:
        """Check if the agent can reach the goal using BFS on the grid.

        This is a pure-Python BFS (no JAX) for validation during generation.
        """
        rows = grid.split('\n')
        h, w = len(rows), len(rows[0])

        # Build passable map
        passable = [[False] * w for _ in range(h)]
        for y in range(h):
            for x in range(w):
                passable[y][x] = (rows[y][x] != '#')

        ax, ay = int(level.agent_pos[0]), int(level.agent_pos[1])
        gx, gy = int(level.goal_pos[0]), int(level.goal_pos[1])

        # BFS
        from collections import deque
        visited = set()
        queue = deque([(ax, ay)])
        visited.add((ax, ay))

        while queue:
            x, y = queue.popleft()
            if x == gx and y == gy:
                return True
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited and passable[ny][nx]:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        return False

    def generate_batch(
        self,
        n: int,
        references: Optional[List[ReferenceMaze]] = None,
        pairwise_metrics: Optional[List[PairwiseMetricEntry]] = None,
        global_metrics: Optional[List[MetricEntry]] = None,
        instruction: str = "",
        target_metrics: Optional[List[MetricEntry]] = None,
        solvability_checker: Optional[Callable] = None,
    ) -> List[GenerationResult]:
        """Generate multiple mazes sequentially.

        Args:
            n: Number of mazes to generate
            (other args same as generate())

        Returns:
            List of GenerationResult objects
        """
        results = []
        for i in range(n):
            logger.info(f"Generating maze {i+1}/{n}...")
            result = self.generate(
                references=references,
                pairwise_metrics=pairwise_metrics,
                global_metrics=global_metrics,
                instruction=instruction,
                target_metrics=target_metrics,
                solvability_checker=solvability_checker,
            )
            results.append(result)

        successes = sum(1 for r in results if r.success)
        logger.info(f"Batch complete: {successes}/{n} successful")
        return results

    def generate_with_feedback(
        self,
        agent_evaluator,
        reference_trajectories: List[Dict],
        reference_labels: List[str],
        references: Optional[List[ReferenceMaze]] = None,
        pairwise_metrics: Optional[List[PairwiseMetricEntry]] = None,
        global_metrics: Optional[List[MetricEntry]] = None,
        instruction: str = "",
        target_metrics: Optional[List[MetricEntry]] = None,
        solvability_checker: Optional[Callable] = None,
        diversity_thresholds=None,
        max_diversity_retries: int = 2,
        n_rollouts: int = 100,
    ) -> GenerationResult:
        """Generate a maze with full metric feedback loop.

        Workflow:
        1. Generate a valid maze (format + solvability)
        2. Run the agent on it to get trajectory
        3. Compute diversity metrics vs reference trajectories
        4. If not diverse enough, send metric feedback to LLM and retry
        5. Repeat up to max_diversity_retries times

        Args:
            agent_evaluator: AgentEvaluator instance for rollouts
            reference_trajectories: List of trajectory dicts from reference mazes
            reference_labels: Labels for reference mazes
            references: Reference mazes for the prompt
            pairwise_metrics: Pairwise metrics for the prompt
            global_metrics: Buffer-wide metrics for the prompt
            instruction: Custom instruction
            target_metrics: Target metric values
            solvability_checker: Optional solvability checker
            diversity_thresholds: DiversityThresholds instance (or None for defaults)
            max_diversity_retries: Max times to retry after diversity gate failure
            n_rollouts: Number of agent rollouts per maze

        Returns:
            GenerationResult with additional gate_result attribute
        """
        from llm.decision_gate import evaluate_candidate, DiversityThresholds
        from llm.prompt_builder import (
            build_diversity_feedback_prompt,
            overlay_path_on_grid,
        )
        from metrics import (
            PositionDTWAnalyzer,
            PolicyEntropyAnalyzer,
            PerStepRegretAnalyzer,
        )

        thresholds = diversity_thresholds or DiversityThresholds()
        start = time.time()

        # Step 1: Generate a structurally valid maze
        result = self.generate(
            references=references,
            pairwise_metrics=pairwise_metrics,
            global_metrics=global_metrics,
            instruction=instruction,
            target_metrics=target_metrics,
            solvability_checker=solvability_checker,
        )

        if not result.success:
            result.latency_ms = (time.time() - start) * 1000
            return result

        # Step 2-4: Metric feedback loop
        for diversity_attempt in range(max_diversity_retries + 1):
            # Run agent on candidate (100 rollouts for robust regret)
            logger.info(f"Running agent on candidate (diversity attempt {diversity_attempt + 1})...")
            candidate_traj = agent_evaluator.evaluate_level_multi_rollout(
                result.level, n_rollouts=n_rollouts,
            )
            solve_rate = candidate_traj.get("solve_rate", 0.0)
            best_return = candidate_traj.get("best_return", 0.0)
            logger.info(
                f"  100-rollout: solve_rate={solve_rate:.0%}, "
                f"best_return={best_return:.3f}"
            )

            # Compute diversity metrics (using best_return for regret)
            gate_result = evaluate_candidate(
                candidate_traj,
                reference_trajectories,
                reference_labels,
                thresholds,
                stored_max_return=best_return,
            )

            # Log metrics
            regret_str = ""
            if gate_result.regret_info is not None:
                ri = gate_result.regret_info
                regret_str = (
                    f", regret={ri.regret:.3f}, "
                    f"ep_len={ri.episode_length}, solved={ri.solved}"
                )
            logger.info(
                f"Diversity gate: {'PASS' if gate_result.accepted else 'FAIL'} — "
                f"min_pos_dtw={gate_result.summary.get('min_pos_dtw', 0):.3f}"
                f"{regret_str}"
            )

            if gate_result.accepted:
                logger.info(f"Maze accepted after {diversity_attempt + 1} diversity check(s)")
                result.gate_metrics = gate_result.summary
                result.gate_pair_metrics = gate_result.pair_metrics
                result.diversity_attempts = diversity_attempt + 1
                break

            if diversity_attempt >= max_diversity_retries:
                logger.info(
                    f"Diversity gate failed after {max_diversity_retries + 1} attempts, "
                    f"accepting anyway"
                )
                result.gate_metrics = gate_result.summary
                result.gate_pair_metrics = gate_result.pair_metrics
                result.diversity_attempts = diversity_attempt + 1
                result.diversity_issues = gate_result.issues
                break

            # Build diversity feedback and regenerate
            logger.info(f"Diversity issues: {gate_result.issues}")

            # Build path overlay for candidate
            candidate_overlay = None
            try:
                positions = candidate_traj["positions"]
                dones = candidate_traj["dones"]
                done_idx = np.where(dones)[0]
                end = done_idx[0] + 1 if len(done_idx) > 0 else len(positions)
                candidate_overlay = overlay_path_on_grid(
                    result.grid, positions[:end]
                )
            except Exception:
                pass

            # Build spatial/causal analysis sections and reference overlays.
            # Only build analyzers for metrics that were actually computed.
            analysis_sections = []
            reference_overlays = {}

            if gate_result.pair_metrics:
                # Position DTW analysis (only if pos DTW was computed)
                pos_pairs = [p for p in gate_result.pair_metrics if p.pos_dtw_path is not None]
                if pos_pairs:
                    most_similar = min(pos_pairs, key=lambda p: p.pos_dtw_distance)
                    ref_idx = reference_labels.index(most_similar.ref_label)
                    ref_traj = reference_trajectories[ref_idx]
                    ref_grid = references[ref_idx].grid if references else None

                    # Reference path overlay for visual comparison
                    if ref_grid:
                        try:
                            ref_pos = ref_traj["positions"]
                            ref_dones = ref_traj["dones"]
                            ref_done_idx = np.where(ref_dones)[0]
                            ref_end = ref_done_idx[0] + 1 if len(ref_done_idx) > 0 else len(ref_pos)
                            reference_overlays[most_similar.ref_label] = overlay_path_on_grid(
                                ref_grid, ref_pos[:ref_end]
                            )
                        except Exception:
                            pass

                    try:
                        pos_analyzer = PositionDTWAnalyzer(
                            candidate_positions=candidate_traj["positions"],
                            candidate_dones=candidate_traj["dones"],
                            reference_positions=ref_traj["positions"],
                            reference_dones=ref_traj["dones"],
                            reference_label=most_similar.ref_label,
                        )
                        analysis_sections.append(pos_analyzer.to_section())
                    except Exception as e:
                        logger.warning(f"Position DTW analysis failed: {e}")

            # Per-step entropy analysis (if entropy was collected)
            if "entropy" in candidate_traj:
                most_similar_label = gate_result.most_similar_ref
                ref_entropy = None
                ref_dones_for_entropy = None
                ref_label_for_entropy = "Reference"
                if most_similar_label and reference_labels:
                    try:
                        ms_idx = reference_labels.index(most_similar_label)
                        ms_traj = reference_trajectories[ms_idx]
                        if "entropy" in ms_traj:
                            ref_entropy = ms_traj["entropy"]
                            ref_dones_for_entropy = ms_traj["dones"]
                            ref_label_for_entropy = most_similar_label
                    except (ValueError, KeyError):
                        pass

                try:
                    entropy_analyzer = PolicyEntropyAnalyzer(
                        candidate_entropy=candidate_traj["entropy"],
                        candidate_dones=candidate_traj["dones"],
                        label="Candidate",
                        reference_entropy=ref_entropy,
                        reference_dones=ref_dones_for_entropy,
                        reference_label=ref_label_for_entropy,
                    )
                    analysis_sections.append(entropy_analyzer.to_section())
                except Exception as e:
                    logger.warning(f"Policy entropy analysis failed: {e}")

            # Per-step regret analysis (always available — uses values/rewards)
            most_similar_label = gate_result.most_similar_ref
            ref_vals_for_regret = None
            ref_rewards_for_regret = None
            ref_dones_for_regret = None
            ref_label_for_regret = "Reference"
            if most_similar_label and reference_labels:
                try:
                    ms_idx = reference_labels.index(most_similar_label)
                    ms_traj = reference_trajectories[ms_idx]
                    ref_vals_for_regret = ms_traj["values"]
                    ref_rewards_for_regret = ms_traj["rewards"]
                    ref_dones_for_regret = ms_traj["dones"]
                    ref_label_for_regret = most_similar_label
                except (ValueError, KeyError):
                    pass

            try:
                regret_analyzer = PerStepRegretAnalyzer(
                    candidate_values=candidate_traj["values"],
                    candidate_rewards=candidate_traj["rewards"],
                    candidate_dones=candidate_traj["dones"],
                    label="Candidate",
                    reference_values=ref_vals_for_regret,
                    reference_rewards=ref_rewards_for_regret,
                    reference_dones=ref_dones_for_regret,
                    reference_label=ref_label_for_regret,
                )
                analysis_sections.append(regret_analyzer.to_section())
            except Exception as e:
                logger.warning(f"Per-step regret analysis failed: {e}")

            feedback_prompt = build_diversity_feedback_prompt(
                result.grid,
                candidate_overlay,
                gate_result.issues,
                analysis_sections=analysis_sections,
                reference_overlays=reference_overlays,
            )

            # Save the feedback prompt and the previous raw responses for debugging
            prev_raw = list(result.raw_responses)
            prev_feedback = list(result.feedback_prompts)

            # Regenerate with feedback context
            result = self.generate(
                references=references,
                pairwise_metrics=pairwise_metrics,
                global_metrics=global_metrics,
                instruction=feedback_prompt,
                target_metrics=target_metrics,
                solvability_checker=solvability_checker,
            )

            # Carry forward the history from previous attempts
            result.raw_responses = prev_raw + result.raw_responses
            result.feedback_prompts = prev_feedback + [feedback_prompt]

            if not result.success:
                break

        result.latency_ms = (time.time() - start) * 1000
        return result

    def generate_batch_with_feedback(
        self,
        n: int,
        agent_evaluator,
        reference_trajectories: List[Dict],
        reference_labels: List[str],
        references: Optional[List[ReferenceMaze]] = None,
        pairwise_metrics: Optional[List[PairwiseMetricEntry]] = None,
        global_metrics: Optional[List[MetricEntry]] = None,
        instruction: str = "",
        target_metrics: Optional[List[MetricEntry]] = None,
        solvability_checker: Optional[Callable] = None,
        diversity_thresholds=None,
        max_diversity_retries: int = 2,
        n_rollouts: int = 100,
    ) -> List[GenerationResult]:
        """Generate multiple mazes with metric feedback loop.

        Args:
            n: Number of mazes to generate
            (other args same as generate_with_feedback())

        Returns:
            List of GenerationResult objects
        """
        results = []
        for i in range(n):
            logger.info(f"Generating maze {i+1}/{n} (with feedback)...")
            result = self.generate_with_feedback(
                agent_evaluator=agent_evaluator,
                reference_trajectories=reference_trajectories,
                reference_labels=reference_labels,
                references=references,
                pairwise_metrics=pairwise_metrics,
                global_metrics=global_metrics,
                instruction=instruction,
                target_metrics=target_metrics,
                solvability_checker=solvability_checker,
                diversity_thresholds=diversity_thresholds,
                max_diversity_retries=max_diversity_retries,
                n_rollouts=n_rollouts,
            )
            results.append(result)

        successes = sum(1 for r in results if r.success)
        logger.info(f"Feedback batch complete: {successes}/{n} successful")
        return results
