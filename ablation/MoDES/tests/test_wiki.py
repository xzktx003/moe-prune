import unittest

from tasks.wiki import normalize_wiki_text, split_wiki_text, wiki_transform


class WikiTaskTest(unittest.TestCase):
    def test_split_wiki_text_keeps_non_empty_completion(self):
        prompt, completion = split_wiki_text(" ".join(f"word{i}" for i in range(80)))

        self.assertTrue(prompt.startswith("word0"))
        self.assertTrue(completion)
        self.assertNotEqual(prompt, completion)

    def test_wiki_transform_builds_text_only_examples(self):
        batch = {
            "text": [
                " ".join(f"token{i}" for i in range(70)),
                "A   short   paragraph with   repeated whitespace that should be normalized "
                + " ".join(f"tail{i}" for i in range(40)),
            ]
        }

        transformed = wiki_transform(batch)

        self.assertEqual(len(transformed["model_input_visual"]), 2)
        self.assertEqual(transformed["model_input_visual"], [None, None])
        self.assertTrue(
            transformed["model_input_text"][0].startswith(
                "Continue the following Wikipedia passage:\n\n"
            )
        )
        self.assertEqual(
            normalize_wiki_text(batch["text"][1]).split()[0],
            transformed["model_input_org_text"][1].split()[0],
        )
        self.assertTrue(transformed["model_input_full_answer"][0])


if __name__ == "__main__":
    unittest.main()
