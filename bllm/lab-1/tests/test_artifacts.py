import json
from datetime import UTC, datetime

import pytest

from catalogue.bootstrap import application
from catalogue.errors import EvidenceError
from catalogue.evidence import EvidenceStore, Journal
from catalogue.persistence.sqlite import populate_catalogue
from catalogue.review.inputs import ReviewInputs
from catalogue.review.models import (
    Experiment,
    ReportTable,
    ReviewContext,
    ReviewResults,
    SectionResult,
)
from catalogue.review.report import build_report, validate_results, verify
from catalogue.review.runner import ReviewRunner


def test_writes_and_journal_redact_nested_secrets(tmp_path):
    secret = "private-artifact-token"
    store = EvidenceStore(tmp_path, (secret,))
    store.write_json("result.json", {"nested": [secret], secret: "key"})
    store.write_text("result.md", secret)
    journal = Journal(
        secrets=(secret,), sink=lambda event: store.append_event("events.jsonl", event)
    )
    journal.record("request", "input_received", nested={"value": secret})
    assert all(secret not in file.read_text() for file in tmp_path.iterdir())
    assert store.read_json("result.json") == {"nested": ["[REDACTED]"], "[REDACTED]": "key"}
    assert secret not in json.dumps(journal.events)
    assert len(list(tmp_path.iterdir())) == 3


def test_io_errors_preserve_cause_and_existing_content(tmp_path):
    target = tmp_path / "result.json"
    target.mkdir()
    store = EvidenceStore(tmp_path)
    with pytest.raises(EvidenceError) as captured:
        store.write_json("result.json", {"answer": 1})
    assert isinstance(captured.value.__cause__, OSError)
    assert target.is_dir()
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("name", ["../file.json", "/file.json", ""])
def test_evidence_cannot_escape_destination(tmp_path, name):
    with pytest.raises(EvidenceError):
        EvidenceStore(tmp_path).write_text(name, "value")


def test_report_refuses_incomplete_results(configure, product_file):
    configure(PRODUCTS_PATH=product_file)
    with application() as services:
        services.store.write_json(
            services.settings.results_filename,
            ReviewResults(started_at=datetime.now(UTC), mode="live", metadata={}).model_dump(
                mode="json"
            ),
        )
        with pytest.raises(EvidenceError, match="incomplete"):
            build_report(services, ReviewInputs.load(services.settings))
        assert not (services.store.directory / services.settings.report_filename).exists()


def test_runner_accepts_injected_experiment_without_changes(configure, product_file):
    configure(PRODUCTS_PATH=product_file)
    with application() as services:
        inputs = ReviewInputs.load(services.settings)
        populate_catalogue(services.settings.database_path, inputs.products)
        inputs.plan = inputs.plan.model_copy(update={"offline_studies": ["new_experiment"]})

        def custom(context):
            context.services.journal.record("custom", "validated")
            return SectionResult(
                name="new_experiment",
                title="Custom experiment",
                source="test",
                records=[{"case_id": "new", "correct": True}],
                expected_cases=["new"],
                table=ReportTable(columns=["Result"], rows=[["Passed"]]),
            )

        runner = ReviewRunner((Experiment("new_experiment", custom),))
        results = runner.run(ReviewContext(services, inputs), live=False)
        assert results.completed_at is not None
        build_report(services, inputs)
        verify(services, inputs)
        assert {path.name for path in services.store.directory.iterdir()} == {
            services.settings.results_filename,
            services.settings.report_filename,
            services.settings.events_filename,
        }
        assert (
            "Custom experiment"
            in (services.store.directory / services.settings.report_filename).read_text()
        )


def test_failed_experiment_keeps_partial_results_and_is_not_reportable(configure, product_file):
    configure(PRODUCTS_PATH=product_file)
    with application() as services:
        inputs = ReviewInputs.load(services.settings)
        populate_catalogue(services.settings.database_path, inputs.products)
        inputs.plan = inputs.plan.model_copy(update={"offline_studies": ["broken"]})

        def broken(context):
            raise TypeError("scenario bug")

        with pytest.raises(TypeError):
            ReviewRunner((Experiment("broken", broken),)).run(
                ReviewContext(services, inputs), False
            )
        saved = services.store.read_json(services.settings.results_filename)
        assert saved["completed_at"] is None
        with pytest.raises(EvidenceError, match="incomplete"):
            build_report(services, inputs)


def test_failed_expectations_are_not_reported_as_success(configure, product_file):
    configure(PRODUCTS_PATH=product_file)
    with application() as services:
        inputs = ReviewInputs.load(services.settings)
        results = ReviewResults(
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            mode="offline",
            metadata={},
            sections=[
                SectionResult(
                    name="failure_handling",
                    title="Failure",
                    source="test",
                    records=[{"case_id": "failed", "correct": False}],
                    expected_cases=["failed"],
                    table=ReportTable(columns=[], rows=[]),
                )
            ],
        )
        with pytest.raises(EvidenceError, match="failed expectations"):
            validate_results(results, inputs)


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, "Да"),
        (False, "Нет"),
        (None, "—"),
        ("", "—"),
        ("first|second\nthird", "first\\|second<br>third"),
    ],
)
def test_report_cells_render_readable_labels_without_changing_evidence(value, expected):
    from catalogue.review.report import cell

    assert cell(value) == expected
