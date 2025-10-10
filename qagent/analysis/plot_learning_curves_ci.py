"""Command-line entry point for manifest-guided learning curve plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from qagent.analysis.plot_learning_curves import plot_learning_curves_ci


def _load_manifest(path: Path) -> List[Dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Manifest must be a JSON list of run entries.")
    return [entry for entry in data if isinstance(entry, dict)]


def _extract_agent_key(run_id: str) -> Optional[str]:
    if "-run" not in run_id:
        return None
    prefix = run_id.rsplit("-run", 1)[0]
    if "-" not in prefix:
        return None
    return prefix.rsplit("-", 1)[1]


def _infer_agent_names(manifest: List[Dict[str, object]], only_agents: Optional[List[str]]) -> Dict[str, str]:
    agent_display: Dict[str, str] = {}
    for entry in manifest:
        run_id = str(entry.get("run_id", ""))
        agent_key = _extract_agent_key(run_id)
        if not agent_key:
            continue
        if only_agents is not None and agent_key not in only_agents:
            continue
        display_name = str(entry.get("architecture", agent_key))
        agent_display.setdefault(agent_key, display_name)
    if only_agents:
        missing = [agent for agent in only_agents if agent not in agent_display]
        if missing:
            raise ValueError(f"Manifest does not contain runs for agents: {', '.join(missing)}")
    if not agent_display:
        raise ValueError("No matching agents found in manifest.")
    return agent_display


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot exploitability curves with 95% CI using the run manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to run_manifest.json produced by build_manifest.py")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("ablation_study_results.json"),
        help="Path to ablation_study_results.json (time-series exploitability data).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination filename for the plotted figure.")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=None,
        help="Optional subset of agent keys to include (must match keys in ablation results).",
    )
    parser.add_argument("--title", type=str, default="Leduc Hold'em: DCFR Exploitability", help="Plot title.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_entries = _load_manifest(args.manifest)
    agent_names = _infer_agent_names(manifest_entries, args.agents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_learning_curves_ci(
        results_file=str(args.results),
        agent_names=agent_names,
        title=args.title,
        output_filename=str(args.output),
    )


if __name__ == "__main__":
    main()
