# E2 files and the Noise v2 protocol

| Name in older runs | Current file |
| --- | --- |
| `noise_conditions_v2.py` | [Shared noise generator](../experiments/support/shared/noise_conditions.py) |
| `run_e2_evaluation_v2.py` | [E2 evaluator](../experiments/e2_noisy_baseline/run_e2_evaluation.py) |
| `run_e2_evaluation_v2_resumable.py` | [E2 entry point](../experiments/e2_noisy_baseline/run_e2_evaluation_resumable.py) |
| `visualize_noise_v2.py` | [Noise preview](../experiments/support/shared/visualize_noise.py) |

Use the [current E2 commands](run_e2.md) for new evaluations. The four older Python filenames in this table identify scripts in previous revisions. The [noise configuration](../configs/noise_conditions_v2.json) still defines the nine noisy test conditions and the clean reference.

The original run records retain their paths and source hashes. E6 checks the noise generator against the source hash in the archived E5 manifest, accounting for the public function alias and the relocated config lookup in the standalone self-test. The [previous verification record](checks/e2_naming_round2/README.md) describes the earlier layout.

New E2 outputs default to `results/local/e2/`. Set `SLEEP_RESULT_DIR` to choose another root or pass `--output-dir` to the resumable evaluator. An existing local `e2_v2/` folder is reused when it is the only E2 folder under that root. E5's reference `c2547fb:result/e2_v2/predictions_with_ids.csv` addresses a file at that historical commit.
