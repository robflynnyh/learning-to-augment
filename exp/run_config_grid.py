import argparse
import shlex
import subprocess
from pathlib import Path

from config_grid import materialize_grid_configs


DEFAULT_WITH_GPU = "/store/store5/software/simple-gpu-schedule/with-gpu"


def quote(value):
    return shlex.quote(str(value))


def build_entrypoint_command(entrypoint, config_path):
    return entrypoint.format(config=quote(config_path), config_raw=str(config_path))


def build_sequential_command(workdir, entrypoint, config_path):
    command = build_entrypoint_command(entrypoint, config_path)
    return f"cd {quote(workdir)} && {command}"


def build_mimas_command(workdir, entrypoint, config_path, with_gpu, gpu_pool):
    inner = build_sequential_command(workdir, entrypoint, config_path)
    return f"{quote(with_gpu)} {quote(gpu_pool)} -- bash -lc {quote(inner)}"


def build_slurm_command(sbatch_script, config_path, export_var, extra_exports, chdir):
    exports = [f"{export_var}={config_path}"]
    exports.extend(extra_exports)
    export_arg = ",".join(exports)
    return f"sbatch --chdir={quote(chdir)} --export={quote(export_arg)} {quote(sbatch_script)}"


def run_shell(command, dry_run):
    print(command)
    if dry_run:
        return 0
    completed = subprocess.run(command, shell=True, check=False)
    return completed.returncode


def launch_grid(args):
    output_dir = args.output_dir
    if output_dir is None:
        source = Path(args.grid_config)
        output_dir = source.parent / ".generated" / source.stem

    specs = materialize_grid_configs(args.grid_config, output_dir)
    print(f"Materialized {len(specs)} configs under {output_dir}")

    if args.materialize_only:
        for spec in specs:
            print(spec["path"])
        return 0

    workdir = Path(args.workdir).resolve()
    sbatch_script = Path(args.sbatch_script).resolve() if args.sbatch_script else None
    commands = []
    for spec in specs:
        config_path = spec["path"].resolve()
        if args.mode == "sequential":
            commands.append(build_sequential_command(workdir, args.entrypoint, config_path))
        elif args.mode == "mimas":
            commands.append(
                build_mimas_command(
                    workdir,
                    args.entrypoint,
                    config_path,
                    args.with_gpu,
                    args.gpu_pool,
                )
            )
        elif args.mode == "slurm":
            if not args.sbatch_script:
                raise ValueError("--sbatch-script is required for --mode slurm")
            commands.append(
                build_slurm_command(
                    sbatch_script,
                    config_path,
                    args.export_var,
                    args.extra_export,
                    args.slurm_chdir or sbatch_script.parent,
                )
            )
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

    if args.jobs_file:
        jobs_file = Path(args.jobs_file)
        jobs_file.parent.mkdir(parents=True, exist_ok=True)
        jobs_file.write_text("\n".join(commands) + "\n")

    if args.mode == "mimas" and args.parallel and not args.dry_run:
        processes = []
        for command in commands:
            print(command)
            processes.append(subprocess.Popen(command, shell=True))
        return max(process.wait() for process in processes)

    status = 0
    for command in commands:
        status = max(status, run_shell(command, args.dry_run))
        if status != 0 and args.stop_on_error:
            return status
    return status


def main():
    parser = argparse.ArgumentParser(
        description="Materialize and run a grid-style experiment config."
    )
    parser.add_argument("--grid-config", required=True, help="YAML containing a top-level grid block")
    parser.add_argument(
        "--entrypoint",
        default="python eval.py --config {config}",
        help="Command template for one materialized config. Use {config} for a shell-quoted path.",
    )
    parser.add_argument(
        "--mode",
        choices=["sequential", "mimas", "slurm"],
        default="sequential",
        help="How to launch materialized configs.",
    )
    parser.add_argument("--workdir", default="exp", help="Directory where the entrypoint should run")
    parser.add_argument("--output-dir", help="Directory for generated one-run YAMLs")
    parser.add_argument("--materialize-only", action="store_true", help="Only write generated YAMLs")
    parser.add_argument("--dry-run", action="store_true", help="Print launch commands without running them")
    parser.add_argument("--jobs-file", help="Optional file to write the generated launch commands")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop sequential launches after a failure")
    parser.add_argument("--parallel", action="store_true", help="Launch Mimas with-gpu commands concurrently")
    parser.add_argument("--with-gpu", default=DEFAULT_WITH_GPU, help="Path to the with-gpu launcher")
    parser.add_argument("--gpu-pool", default="1,2", help="GPU pool passed to with-gpu")
    parser.add_argument("--sbatch-script", help="Slurm batch script used in --mode slurm")
    parser.add_argument(
        "--slurm-chdir",
        help="Working directory for sbatch. Defaults to the sbatch script directory.",
    )
    parser.add_argument("--export-var", default="CONFIG", help="Slurm export variable for the config path")
    parser.add_argument(
        "--extra-export",
        action="append",
        default=[],
        help="Additional KEY=VALUE entries for sbatch --export. Can be repeated.",
    )
    args = parser.parse_args()
    raise SystemExit(launch_grid(args))


if __name__ == "__main__":
    main()
