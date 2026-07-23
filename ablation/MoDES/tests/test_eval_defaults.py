import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def class_method_names(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name for child in node.body if isinstance(child, ast.FunctionDef)
            }
    raise AssertionError(f"Class {class_name} not found in {path}")


class EvalDefaultConfigTest(unittest.TestCase):
    def test_kimi_eval_sets_default_max_length_to_2048(self):
        text = (ROOT / "eval" / "kimi.py").read_text()
        self.assertIn('self._max_length = kwargs.get("max_length", 2048)', text)
        self.assertIn("def max_length", text)

    def test_qwen3_eval_keeps_default_max_length_to_2048(self):
        text = (ROOT / "eval" / "qwen3.py").read_text()
        self.assertIn('self._max_length = kwargs.get("max_length", 2048)', text)

    def test_eval_wrappers_expose_max_length_property(self):
        self.assertIn(
            "max_length", class_method_names(ROOT / "eval" / "qwen3.py", "Qwen3_VL")
        )
        self.assertIn(
            "max_length", class_method_names(ROOT / "eval" / "kimi.py", "KimiVL")
        )


if __name__ == "__main__":
    unittest.main()
