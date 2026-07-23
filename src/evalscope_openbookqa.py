"""EvalScope benchmark adapter for OpenBookQA (main split).

OpenBookQA is not one of evalscope's built-in benchmarks, so we register a
``MultiChoiceAdapter`` that pulls ``allenai/openbookqa`` from the hub.

Dataset fields:
    - ``question_stem`` (str): the question text
    - ``choices`` ({label, text}): 4 choices, labels ``A..D``
    - ``answerKey`` (str): one of ``A..D``
    - ``id`` (str)

Import this module once to register the benchmark under the name
``openbookqa``::

    import moe_prune.code.src.evalscope_openbookqa  # noqa: F401
"""

from __future__ import annotations

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='openbookqa',
        pretty_name='OpenBookQA',
        tags=[Tags.REASONING, Tags.MULTIPLE_CHOICE],
        description=(
            'OpenBookQA is a QA dataset that probes understanding of elementary '
            'science facts via 4-way multiple choice. This adapter targets the '
            '``main`` config of ``allenai/openbookqa``.'
        ),
        dataset_id='allenai/openbookqa',
        subset_list=['main'],
        metric_list=['acc'],
        few_shot_num=0,
        train_split='train',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class OpenBookQAAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record) -> Sample:
        choices_field = record['choices']
        if isinstance(choices_field, dict):
            choice_texts = list(choices_field['text'])
        else:
            # Fallback for list-of-dict layout.
            choice_texts = [c['text'] for c in choices_field]

        return Sample(
            input=record['question_stem'],
            choices=choice_texts,
            target=record['answerKey'],
            metadata={'id': record.get('id', '')},
        )
