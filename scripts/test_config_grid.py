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

    def test_cases_can_product_with_axes_and_labeled_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "grid.yaml"
            output_dir = Path(tmp) / "generated"
            config_path.write_text(
                """
evaluation:
  search_repeats: 1
  optim_args:
    lr: 1e-6
    single_step_lr: 4e-2
  save_path: "results/lr{grid_label:evaluation.optim_args.lr}_searchlr{grid_label:evaluation.optim_args.single_step_lr}.txt"
grid:
  name: paired
  combine: product
  id_template: "lr{grid_label:evaluation.optim_args.lr}_searchlr{grid_label:evaluation.optim_args.single_step_lr}_repeats{evaluation.search_repeats}"
  axes:
    evaluation.search_repeats: [1, 2]
  cases:
    - id: historical
      values:
        evaluation.optim_args.lr:
          value: 1e-6
          label: "1e-6"
        evaluation.optim_args.single_step_lr:
          value: 4e-2
          label: "4e-2"
    - id: recent
      values:
        evaluation.optim_args.lr:
          value: 8e-6
          label: "8e-6"
        evaluation.optim_args.single_step_lr:
          value: 9e-2
          label: "9e-2"
""".lstrip()
            )

            specs = materialize_grid_configs(config_path, output_dir)
            recent = OmegaConf.load(output_dir / "lr8e-6_searchlr9e-2_repeats2.yaml")

            self.assertEqual(len(specs), 4)
            self.assertTrue((output_dir / "lr1e-6_searchlr4e-2_repeats1.yaml").exists())
            self.assertTrue((output_dir / "lr8e-6_searchlr9e-2_repeats2.yaml").exists())
        self.assertEqual(recent.evaluation.search_repeats, 2)
        self.assertEqual(recent.evaluation.optim_args.lr, 8e-6)
        self.assertTrue(recent.evaluation.save_path.endswith("lr8e-6_searchlr9e-2.txt"))


if __name__ == "__main__":
    unittest.main()
