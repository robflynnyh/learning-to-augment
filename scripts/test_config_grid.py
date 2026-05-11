import tempfile
import unittest
from pathlib import Path
import sys

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp.config_grid import expand_grid_config, materialize_grid_configs


class ConfigGridTests(unittest.TestCase):
    def test_axes_expand_to_cartesian_product(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "grid.yaml"
            config_path.write_text(
                """
checkpointing:
  asr_model: ckpt.pt
evaluation:
  search_repeats: 1
  optim_args:
    lr: 1e-6
  save_path: "results/{grid_id}.txt"
grid:
  name: example
  id_template: "repeats_{evaluation.search_repeats}_lr_{evaluation.optim_args.lr}"
  id_path: evaluation.id
  axes:
    evaluation.search_repeats: [1, 2]
    evaluation.optim_args.lr: [1e-6, 2e-6]
""".lstrip()
            )

            specs = expand_grid_config(config_path)

        self.assertEqual(len(specs), 4)
        ids = [spec["id"] for spec in specs]
        self.assertIn("repeats_1_lr_1e-06", ids)
        first = specs[0]["config"]
        self.assertEqual(first.evaluation.id, specs[0]["id"])
        self.assertTrue(first.evaluation.save_path.endswith(".txt"))
        self.assertNotIn("grid", first)

    def test_cases_can_repeat_or_override_individual_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "grid.yaml"
            output_dir = Path(tmp) / "generated"
            config_path.write_text(
                """
evaluation:
  search_repeats: 5
  save_path: base.txt
grid:
  name: cases
  cases:
    - id: no_save
      values:
        evaluation.save_path: ""
    - id: saved_repeat
      evaluation.search_repeats: 10
""".lstrip()
            )

            specs = materialize_grid_configs(config_path, output_dir)
            no_save = OmegaConf.load(output_dir / "no_save.yaml")
            saved_repeat = OmegaConf.load(output_dir / "saved_repeat.yaml")

        self.assertEqual(len(specs), 2)
        self.assertEqual(no_save.evaluation.save_path, "")
        self.assertEqual(saved_repeat.evaluation.search_repeats, 10)


if __name__ == "__main__":
    unittest.main()
