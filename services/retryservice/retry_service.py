#!/usr/bin/env python3
import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path


logger = logging.getLogger("retry_service")


class JsonlHandler(logging.Handler):
    """Разделяет наблюдаемые логи и скрытый эталон вызовов."""

    def __init__(self, logs_path: Path, oracle_path: Path):
        super().__init__()
        self.logs = logs_path.open("w", encoding="utf-8")
        self.oracle = oracle_path.open("w", encoding="utf-8")
        self.execution_id = None
        self.event_index = 0

    def begin_execution(self, execution_id: str):
        self.execution_id = execution_id
        self.event_index = 0

    def emit(self, record):
        self.event_index += 1

        observed = {
            "timestamp": datetime.fromtimestamp(
                record.created, timezone.utc
            ).isoformat(),
            "service": "retry_service",
            "execution_id": self.execution_id,
            "event_index": self.event_index,
            "level": record.levelname,
            "message": record.getMessage(),
        }

        oracle = {
            "execution_id": self.execution_id,
            "event_index": self.event_index,
            "event": record.getMessage(),
            "file": Path(record.pathname).name,
            "function": record.funcName,
            "line": record.lineno,
            "call_site_id": (
                f"{Path(record.pathname).name}:"
                f"{record.funcName}:{record.lineno}"
            ),
        }

        self.logs.write(json.dumps(observed) + "\n")
        self.oracle.write(json.dumps(oracle) + "\n")

    def close(self):
        try:
            self.logs.close()
            self.oracle.close()
        finally:
            super().close()


class TemporaryFailure(Exception):
    pass


class RetryService:
    def dependency_call(self, attempt: int, failures_before_success: int):
        if attempt <= failures_before_success:
            raise TemporaryFailure("controlled dependency failure")

    def execute(self, failures_before_success: int, max_attempts: int = 3):
        logger.info("operation started")
        started = time.perf_counter()

        for attempt in range(1, max_attempts + 1):
            logger.info("attempt started")

            try:
                self.dependency_call(attempt, failures_before_success)
            except TemporaryFailure:
                if attempt < max_attempts:
                    logger.warning("temporary failure, retrying")
                    continue

                logger.error("retry budget exhausted")
                outcome = "failed"
                break
            else:
                logger.info("operation succeeded")
                outcome = "success"
                break

        logger.info("operation finished")

        return {
            "outcome": outcome,
            "attempts": attempt,
            "duration_ms": (time.perf_counter() - started) * 1000,
        }


SCENARIOS = {
    "immediate_success": 0,
    "one_retry_success": 1,
    "two_retries_success": 2,
    "retry_exhausted": 3,
}


def build_dataset(output: Path, repetitions: int):
    output.mkdir(parents=True, exist_ok=True)

    handler = JsonlHandler(
        output / "logs.jsonl",
        output / "oracle_events.jsonl",
    )

    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)

    service = RetryService()

    try:
        with (output / "executions.jsonl").open(
            "w", encoding="utf-8"
        ) as manifest:
            for scenario, failures in SCENARIOS.items():
                for repetition in range(repetitions):
                    execution_id = f"{scenario}-{repetition:04d}"
                    handler.begin_execution(execution_id)

                    result = service.execute(failures)

                    # В train есть RETRY, но нет двух повторов и исчерпания.
                    split = (
                        "train"
                        if scenario in {
                            "immediate_success",
                            "one_retry_success",
                        }
                        else "test"
                    )

                    metadata = {
                        "execution_id": execution_id,
                        "scenario": scenario,
                        "split": split,
                        "expected_valid": True,
                        "failures_before_success": failures,
                        **result,
                    }

                    manifest.write(json.dumps(metadata) + "\n")
    finally:
        logger.removeHandler(handler)
        handler.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("fixture_data"))
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()

    if args.repetitions < 1:
        parser.error("--repetitions must be positive")

    build_dataset(args.output, args.repetitions)


if __name__ == "__main__":
    main()