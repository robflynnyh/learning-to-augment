import itertools
import re
from copy import deepcopy
from pathlib import Path

from omegaconf import OmegaConf


GRID_KEY = "grid"
RESERVED_CASE_KEYS = {"id", "label", "values"}


def _as_plain_container(config):
    return OmegaConf.to_container(config, resolve=False)


def _sanitize(value):
    text = str(value)
    text = text.replace("/", "_").replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)
    return text.strip("_") or "value"


def _set_by_path(config, dotted_path, value):
    OmegaConf.update(config, dotted_path, value, merge=False)


def _get_by_path(config, dotted_path, default=""):
    try:
        return OmegaConf.select(config, dotted_path)
    except Exception:
        return default


def _render_template(template, config, run_id, index):
    def replace(match):
        key = match.group(1)
        if key == "grid_id":
            return run_id
        if key == "grid_index":
            return str(index)
        return str(_get_by_path(config, key, match.group(0)))

    return re.sub(r"\{([^{}]+)\}", replace, template)


def _render_strings(value, config, run_id, index):
    if isinstance(value, str):
        return _render_template(value, config, run_id, index)
    if isinstance(value, list):
        return [_render_strings(item, config, run_id, index) for item in value]
    if isinstance(value, dict):
        return {
            key: _render_strings(item, config, run_id, index)
            for key, item in value.items()
        }
    return value


def _axis_combinations(axes):
    if not axes:
        return []
    plain_axes = _as_plain_container(axes)
    keys = list(plain_axes.keys())
    values = []
    for key in keys:
        axis_values = plain_axes[key]
        if not isinstance(axis_values, list) or len(axis_values) == 0:
            raise ValueError(f"grid.axes.{key} must be a non-empty list")
        values.append(axis_values)

    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def _case_combinations(cases):
    if not cases:
        return []
    plain_cases = _as_plain_container(cases)
    for case in plain_cases:
        if not isinstance(case, dict):
            raise ValueError("Every grid.cases entry must be a mapping")
        values = deepcopy(case.get("values", {}))
        for key, value in case.items():
            if key not in RESERVED_CASE_KEYS:
                values[key] = value
        yield case.get("id") or case.get("label"), values


def expand_grid_config(path):
    source_path = Path(path)
    config = OmegaConf.load(source_path)
    if GRID_KEY not in config:
        raise ValueError(f"{source_path} does not contain a top-level '{GRID_KEY}' block")

    grid = config[GRID_KEY]
    base_container = _as_plain_container(config)
    base_container.pop(GRID_KEY, None)

    run_specs = []
    axis_runs = [(None, values) for values in _axis_combinations(grid.get("axes"))]
    case_runs = list(_case_combinations(grid.get("cases")))
    runs = axis_runs + case_runs
    if not runs:
        raise ValueError("Grid config must define grid.axes or grid.cases")

    id_template = grid.get("id_template")
    id_path = grid.get("id_path")
    grid_name = grid.get("name", source_path.stem)

    for index, (case_id, values) in enumerate(runs, start=1):
        run_config = OmegaConf.create(deepcopy(base_container))
        for dotted_path, value in values.items():
            _set_by_path(run_config, dotted_path, value)

        if case_id:
            run_id = str(case_id)
        elif id_template:
            run_id = _render_template(str(id_template), run_config, "", index)
        else:
            parts = [f"{key}={_sanitize(value)}" for key, value in values.items()]
            run_id = "__".join(parts)
        run_id = _sanitize(run_id)

        if id_path:
            _set_by_path(run_config, str(id_path), run_id)

        run_config["grid_metadata"] = {
            "name": grid_name,
            "id": run_id,
            "index": index,
            "source": str(source_path),
            "parameters": values,
        }

        rendered_container = _render_strings(
            _as_plain_container(run_config),
            run_config,
            run_id,
            index,
        )
        run_config = OmegaConf.create(rendered_container)
        run_specs.append(
            {
                "id": run_id,
                "index": index,
                "parameters": values,
                "config": run_config,
            }
        )

    return run_specs


def materialize_grid_configs(path, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    materialized = []
    for spec in expand_grid_config(path):
        config_path = output_path / f"{spec['id']}.yaml"
        OmegaConf.save(spec["config"], config_path)
        materialized.append({**spec, "path": config_path})
    return materialized
