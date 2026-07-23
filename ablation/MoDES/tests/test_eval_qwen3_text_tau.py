import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Qwen3TextTauEvalProtocolTest(unittest.TestCase):
    def test_ppl_eval_uses_raw_wikitext_text(self):
        text = (ROOT / "eval_qwen3_text_tau.py").read_text(encoding="utf-8")

        self.assertIn('parser.add_argument("--min_text_length", type=int, default=512)', text)
        self.assertIn('if len(text) < min_text_length:', text)
        self.assertIn('texts.append(" \\n" if text == "" else text)', text)
        self.assertIn('default="storage/search/modes_text_tau_metrics_wikitext.json"', text)
        self.assertNotIn("apply_chat_template", text)
        self.assertNotIn("Continue the following Wikipedia passage", text)


if __name__ == "__main__":
    unittest.main()
