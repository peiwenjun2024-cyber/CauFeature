"""Route the frozen Table 9 evaluator to freshly generated case inputs."""

from pathlib import Path


FROZEN_EVALUATOR = Path(__file__).with_name("historical") / "frozen_evaluator.py"
EVALUATOR_PATH = Path("code_sensitivity_v6_20261008/run_scaled_evaluator_full_vs_selected.py")


def evaluate(worker, dataset, seed, scorer):
    """Run the historical algorithm against the fresh manifest and artifacts."""
    source = FROZEN_EVALUATOR.resolve(strict=True)
    view = worker.OUT / "source_view"
    link = view / EVALUATOR_PATH
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(source)
    except FileExistsError:
        if not link.is_symlink() or link.resolve(strict=True) != source:
            raise ValueError(f"Unexpected frozen evaluator in the source view: {link}")

    worker.ROOT = view
    # Fresh runs can reuse the two verified RQ1 references in their own manifest.
    # They have no legacy campaign receipts to consult.
    worker.jobs = lambda: ()
    worker.evaluate(dataset, seed, scorer)
