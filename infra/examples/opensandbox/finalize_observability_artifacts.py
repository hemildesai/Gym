# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Finalize sandbox observability artifacts after standalone eval jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nemo_gym.sandbox.observability.recorder import log_wandb_artifact
from nemo_gym.sandbox.observability.render import render_reports
from nemo_gym.sandbox.observability.summary import write_summary
from nemo_gym.sandbox.observability.traces import export_trace_artifacts


def finalize_observability(
    output_dir: Path,
    *,
    run_id: str | None,
    service_name: str,
    max_rendered_trajectories: int,
    render_html: bool,
    render_png: bool,
    export_otlp_json: bool,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_run_name: str | None,
    wandb_artifact_name: str,
    wandb_log_artifact: bool,
    wandb_log_metrics: bool,
    wandb_log_time_series: bool,
    wandb_max_time_series_points: int,
    wandb_log_reports: bool,
    wandb_max_media_items: int,
    wandb_log_tables: bool,
    wandb_max_table_rows: int,
    wandb_metric_prefix: str,
) -> dict[str, object]:
    """Write summaries/reports/traces and optionally upload to W&B."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = write_summary(output_dir)
    trace_outputs: dict[str, str] = {}
    if export_otlp_json:
        export_trace_artifacts(output_dir, service_name=service_name, run_id=run_id)
        trace_outputs = {
            "chrome_trace": str(output_dir / "traces" / "chrome_trace.json"),
            "otel_traces": str(output_dir / "traces" / "otel_traces.json"),
        }
    render_outputs = render_reports(
        output_dir,
        max_rendered_trajectories=max_rendered_trajectories,
        render_html=render_html,
        render_png=render_png,
    )
    if wandb_enabled:
        log_wandb_artifact(
            output_dir,
            {
                "enabled": True,
                "artifact_name": wandb_artifact_name,
                "project": wandb_project,
                "entity": wandb_entity,
                "run_name": wandb_run_name,
                "log_artifact": wandb_log_artifact,
                "log_metrics": wandb_log_metrics,
                "log_time_series": wandb_log_time_series,
                "max_time_series_points": wandb_max_time_series_points,
                "log_reports": wandb_log_reports,
                "max_media_items": wandb_max_media_items,
                "log_tables": wandb_log_tables,
                "max_table_rows": wandb_max_table_rows,
                "metric_prefix": wandb_metric_prefix,
            },
        )
    result = {
        "output_dir": str(output_dir),
        "run_id": run_id,
        "summary_json": str(output_dir / "summary.json"),
        "reports_index": str(output_dir / "reports" / "index.html"),
        "trace_outputs": trace_outputs,
        "render_outputs": render_outputs,
        "wandb_json": str(output_dir / "wandb.json")
        if (output_dir / "wandb.json").exists()
        else None,
        "summary": summary,
    }
    (output_dir / "finalize_observability.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--service-name", default="nemo-gym-sandbox-eval")
    parser.add_argument("--max-rendered-trajectories", type=int, default=100)
    parser.add_argument("--render-html", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-png", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-otlp-json", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", default="nemo-gym-sandbox-eval")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-artifact-name", default="sandbox-observability")
    parser.add_argument("--wandb-log-artifact", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-log-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-log-time-series", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-max-time-series-points", type=int, default=20000)
    parser.add_argument("--wandb-log-reports", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-max-media-items", type=int, default=40)
    parser.add_argument("--wandb-log-tables", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-max-table-rows", type=int, default=20000)
    parser.add_argument("--wandb-metric-prefix", default="sandbox/observability")
    args = parser.parse_args()

    result = finalize_observability(
        args.output_dir,
        run_id=args.run_id,
        service_name=args.service_name,
        max_rendered_trajectories=args.max_rendered_trajectories,
        render_html=args.render_html,
        render_png=args.render_png,
        export_otlp_json=args.export_otlp_json,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_artifact_name=args.wandb_artifact_name,
        wandb_log_artifact=args.wandb_log_artifact,
        wandb_log_metrics=args.wandb_log_metrics,
        wandb_log_time_series=args.wandb_log_time_series,
        wandb_max_time_series_points=args.wandb_max_time_series_points,
        wandb_log_reports=args.wandb_log_reports,
        wandb_max_media_items=args.wandb_max_media_items,
        wandb_log_tables=args.wandb_log_tables,
        wandb_max_table_rows=args.wandb_max_table_rows,
        wandb_metric_prefix=args.wandb_metric_prefix,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
