"""pragma evaluation framework.

Lightweight test-harness for measuring answer quality, retrieval correctness,
and token efficiency. Designed to run offline against any KnowledgeBase
without external dependencies.

Example::

    from pragma.eval import Evaluator, TestCase

    cases = [
        TestCase(
            query="Who founded Apple?",
            expected_answer_contains=["Steve Jobs"],
            expected_entities=["Apple", "Steve Jobs"],
        ),
    ]
    report = Evaluator(kb, cases).run()
    print(report.summary())
"""

from pragma.eval.evaluator import EvalReport, Evaluator, TestCase

__all__ = ["EvalReport", "Evaluator", "TestCase"]
