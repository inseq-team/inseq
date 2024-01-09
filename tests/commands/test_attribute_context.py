import pytest
from pytest import fixture
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, GPT2LMHeadModel, MarianMTModel

from inseq.commands.attribute_context import AttributeContextArgs, AttributeContextOutput, CCIOutput, attribute_context


@fixture(scope="session")
def encdec_model() -> MarianMTModel:
    return AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")


@fixture(scope="session")
def deconly_model() -> GPT2LMHeadModel:
    return AutoModelForCausalLM.from_pretrained("gpt2")


def round_scores(cli_out: AttributeContextOutput) -> AttributeContextOutput:
    cli_out.cti_scores = [round(score, 2) for score in cli_out.cti_scores]
    for idx in range(len(cli_out.cci_scores)):
        cci = cli_out.cci_scores[idx]
        cli_out.cci_scores[idx].cti_score = round(cli_out.cci_scores[idx].cti_score, 2)
        if cci.input_context_scores is not None:
            cli_out.cci_scores[idx].input_context_scores = [round(s, 2) for s in cci.input_context_scores]
        if cci.output_context_scores is not None:
            cli_out.cci_scores[idx].output_context_scores = [round(s, 2) for s in cci.output_context_scores]
    return cli_out


def test_in_out_ctx_encdec_whitespace_sep(encdec_model: MarianMTModel):
    # Base case for context-aware encoder-decoder: no language tag, no special token in separator
    # source context (input context) is translated into target context (output context).
    in_out_ctx_encdec_whitespace_sep = AttributeContextArgs(
        model_name_or_path=encdec_model,
        input_context_text="The girls were away.",
        input_current_text="Where are they?",
        output_template="{context} {current}",
        input_template="{context} {current}",
        attributed_fn="contrast_prob_diff",
        show_viz=False,
        # Pre-defining natural model outputs to avoid user input in unit tests
        output_context_text="",
        output_current_text="Où sont-elles?",
        add_output_info=False,
    )
    expected_output = AttributeContextOutput(
        input_context="The girls were away.",
        input_context_tokens=["▁The", "▁girls", "▁were", "▁away", "."],
        output_context="",
        output_context_tokens=[],
        output_current="Où sont-elles?",
        output_current_tokens=["▁Où", "▁sont", "-", "elles", "?"],
        cti_scores=[1.36, 0.08, 0.34, 1.23, 0.27],
        cci_scores=[
            CCIOutput(
                cti_idx=0,
                cti_token="▁Où",
                cti_score=1.36,
                contextual_prefix="Où",
                contextless_prefix="Où",
                input_context_scores=[0.01, 0.01, 0.01, 0.01, 0.01],
                output_context_scores=[],
            ),
            CCIOutput(
                cti_idx=3,
                cti_token="elles",
                cti_score=1.23,
                contextual_prefix="Où sont-elles",
                contextless_prefix="Où sont-ils",
                input_context_scores=[0.03, 0.12, 0.03, 0.03, 0.02],
                output_context_scores=[],
            ),
        ],
        info=None,
    )
    cli_out = attribute_context(in_out_ctx_encdec_whitespace_sep)
    assert round_scores(cli_out) == expected_output


def test_in_ctx_deconly(deconly_model: GPT2LMHeadModel):
    # Base case for context-aware decoder-only model with input-only context.
    in_ctx_deconly = AttributeContextArgs(
        model_name_or_path=deconly_model,
        input_context_text="George was sick yesterday.",
        input_current_text="His colleagues asked him to come",
        attributed_fn="contrast_prob_diff",
        show_viz=False,
        add_output_info=False,
    )
    expected_output = AttributeContextOutput(
        input_context="George was sick yesterday.",
        input_context_tokens=["George", "Ġwas", "Ġsick", "Ġyesterday", "."],
        output_context=None,
        output_context_tokens=None,
        output_current="to the hospital. He said he was fine",
        output_current_tokens=["to", "Ġthe", "Ġhospital", ".", "ĠHe", "Ġsaid", "Ġhe", "Ġwas", "Ġfine"],
        cti_scores=[0.31, 0.25, 0.55, 0.16, 0.43, 0.19, 0.13, 0.07, 0.37],
        cci_scores=[
            CCIOutput(
                cti_idx=2,
                cti_token="Ġhospital",
                cti_score=0.55,
                contextual_prefix="George was sick yesterday. His colleagues asked him to come to the hospital",
                contextless_prefix="His colleagues asked him to come to the office",
                input_context_scores=[0.39, 0.29, 0.52, 0.26, 0.16],
                output_context_scores=None,
            )
        ],
        info=None,
    )
    cli_out = attribute_context(in_ctx_deconly)
    assert round_scores(cli_out) == expected_output


def test_out_ctx_deconly(deconly_model: GPT2LMHeadModel):
    # Base case for context-aware decoder-only model with forced output context mocking a reasoning chain.
    out_ctx_deconly = AttributeContextArgs(
        model_name_or_path=deconly_model,
        output_template="\n\nLet's think step by step:\n{context}\n\nAnswer:\n{current}",
        input_template="{current}",
        input_current_text="Question: How many pairs of legs do 10 horses have?",
        output_context_text="1. A horse has 4 legs.\n2. 10 horses have 40 legs.\n3. 40 legs make 20 pairs of legs.",
        output_current_text="20 pairs of legs.",
        attributed_fn="contrast_prob_diff",
        show_viz=False,
        add_output_info=False,
    )
    expected_output = AttributeContextOutput(
        input_context=None,
        input_context_tokens=None,
        output_context="1. A horse has 4 legs.\n2. 10 horses have 40 legs.\n3. 40 legs make 20 pairs of legs.",
        output_context_tokens=[
            "1",
            ".",
            "ĠA",
            "Ġhorse",
            "Ġhas",
            "Ġ4",
            "Ġlegs",
            ".",
            "Ċ",
            "2",
            ".",
            "Ġ10",
            "Ġhorses",
            "Ġhave",
            "Ġ40",
            "Ġlegs",
            ".",
            "Ċ",
            "3",
            ".",
            "Ġ40",
            "Ġlegs",
            "Ġmake",
            "Ġ20",
            "Ġpairs",
            "Ġof",
            "Ġlegs",
            ".",
        ],
        output_current="20 pairs of legs.",
        output_current_tokens=["20", "Ġpairs", "Ġof", "Ġlegs", "."],
        cti_scores=[4.53, 1.33, 0.43, 0.74, 0.93],
        cci_scores=[
            CCIOutput(
                cti_idx=0,
                cti_token="20 → Ġ20",
                cti_score=4.53,
                contextual_prefix="Question: How many pairs of legs do 10 horses have?\n\nLet's think step by step:\n1. A horse has 4 legs.\n2. 10 horses have 40 legs.\n3. 40 legs make 20 pairs of legs.\n\nAnswer:\n20",
                contextless_prefix="Question: How many pairs of legs do 10 horses have?\n",
                input_context_scores=None,
                output_context_scores=[0.0] * 28,
            ),
        ],
        info=None,
    )
    cli_out = attribute_context(out_ctx_deconly)
    assert round_scores(cli_out).cci_scores[0] == expected_output.cci_scores[0]


def test_in_out_ctx_deconly(deconly_model: GPT2LMHeadModel):
    # Base case for context-aware decoder-only model with input and forced output context.
    in_out_ctx_deconly = AttributeContextArgs(
        model_name_or_path=deconly_model,
        input_context_text="George was sick yesterday.",
        input_current_text="His colleagues asked him if",
        output_context_text="something was wrong. He said",
        attributed_fn="contrast_prob_diff",
        show_viz=False,
        add_output_info=False,
    )
    expected_output = AttributeContextOutput(
        input_context="George was sick yesterday.",
        input_context_tokens=["George", "Ġwas", "Ġsick", "Ġyesterday", "."],
        output_context="something was wrong. He said",
        output_context_tokens=["something", "Ġwas", "Ġwrong", ".", "ĠHe", "Ġsaid"],
        output_current="he was fine.",
        output_current_tokens=["he", "Ġwas", "Ġfine", "."],
        cti_scores=[1.2, 0.72, 1.5, 0.49],
        cci_scores=[
            CCIOutput(
                cti_idx=2,
                cti_token="Ġfine",
                cti_score=1.5,
                contextual_prefix="George was sick yesterday. His colleagues asked him if something was wrong. He said he was fine",
                contextless_prefix="His colleagues asked him if he was a",
                input_context_scores=[0.19, 0.15, 0.33, 0.13, 0.15],
                output_context_scores=[0.08, 0.07, 0.14, 0.12, 0.09, 0.14],
            )
        ],
        info=None,
    )
    cli_out = attribute_context(in_out_ctx_deconly)
    assert round_scores(cli_out) == expected_output


def test_in_ctx_encdec_special_sep():
    # Encoder-decoder model with special separator tags in input only, context is given only in the source
    # (input context) but not produced in the target (output context).
    in_ctx_encdec_special_sep = AttributeContextArgs(
        model_name_or_path="context-mt/scat-marian-small-ctx4-cwd1-en-fr",
        input_context_text="The girls were away.",
        input_current_text="Where are they?",
        output_template="{current}",
        input_template="{context} <brk> {current}",
        special_tokens_to_keep=["<brk>"],
        attributed_fn="contrast_prob_diff",
        show_viz=False,
        add_output_info=False,
    )
    expected_output = AttributeContextOutput(
        input_context="The girls were away.",
        input_context_tokens=["▁The", "▁girls", "▁were", "▁away", "."],
        output_context=None,
        output_context_tokens=None,
        output_current="Où sont-elles ?",
        output_current_tokens=["▁Où", "▁sont", "-", "elles", "▁?"],
        cti_scores=[0.08, 0.04, 0.01, 0.32, 0.06],
        cci_scores=[
            CCIOutput(
                cti_idx=3,
                cti_token="elles",
                cti_score=0.32,
                contextual_prefix="Où sont-elles",
                contextless_prefix="Où sont-elles",
                input_context_scores=[0.0, 0.0, 0.0, 0.0, 0.0],
                output_context_scores=None,
            )
        ],
        info=None,
    )
    cli_out = attribute_context(in_ctx_encdec_special_sep)
    assert round_scores(cli_out) == expected_output


def test_in_out_ctx_encdec_special_sep():
    # Encoder-decoder model with special separator tags in input and output, context is given in the source (input context)
    # and produced in the target (output context) before the special token separator.
    in_out_ctx_encdec_special_sep = AttributeContextArgs(
        model_name_or_path="context-mt/scat-marian-small-target-ctx4-cwd0-en-fr",
        input_context_text="The girls were away.",
        input_current_text="Where are they?",
        output_template="{context}<brk> {current}",
        input_template="{context} <brk> {current}",
        special_tokens_to_keep=["<brk>"],
        attributed_fn="contrast_prob_diff",
        show_viz=False,
        add_output_info=False,
        # Pre-defining natural model outputs to avoid user input in unit tests
        output_context_text="Les filles étaient parties.",
    )
    expected_output = AttributeContextOutput(
        input_context="The girls were away.",
        input_context_tokens=["▁The", "▁girls", "▁were", "▁away", "."],
        output_context="Les filles étaient parties.",
        output_context_tokens=["▁Les", "▁filles", "▁étaient", "▁parties", "."],
        output_current="Où sont-elles ?",
        output_current_tokens=["▁Où", "▁sont", "-", "elles", "▁?"],
        cti_scores=[0.17, 0.03, 0.02, 3.99, 0.0],
        cci_scores=[
            CCIOutput(
                cti_idx=3,
                cti_token="elles",
                cti_score=3.99,
                contextual_prefix="Les filles étaient parties.<brk>  Où sont-elles",
                contextless_prefix="Où sont-ils",
                input_context_scores=[0.0, 0.0, 0.0, 0.0, 0.0],
                output_context_scores=[0.0] * 5,
            )
        ],
    )
    cli_out = attribute_context(in_out_ctx_encdec_special_sep)
    assert round_scores(cli_out) == expected_output


@pytest.mark.slow
def test_in_out_ctx_encdec_langtag_whitespace_sep():
    # Base case for context-aware encoder-decoder model with language tag in input and output.
    # Context is given in the source (input context) and translated into target context (output context)
    in_out_ctx_encdec_langtag_whitespace_sep = AttributeContextArgs(
        model_name_or_path="facebook/mbart-large-50-one-to-many-mmt",
        input_context_text="The girls were away.",
        input_current_text="Where are they?",
        output_template="{context} {current}",
        input_template="{context} {current}",
        tokenizer_kwargs={"src_lang": "en_XX", "tgt_lang": "fr_XX"},
        attributed_fn="contrast_prob_diff",
        show_viz=False,
        add_output_info=False,
        show_intermediate_outputs=False,
        # Pre-defining natural model outputs to avoid user input in unit tests
        output_context_text="Les filles étaient loin.",
    )
    expected_output = AttributeContextOutput(
        input_context="The girls were away.",
        input_context_tokens=["▁The", "▁girls", "▁were", "▁away", "."],
        output_context="Les filles étaient loin.",
        output_context_tokens=["▁Les", "▁filles", "▁étaient", "▁loin", "."],
        output_current="Où sont-elles?",
        output_current_tokens=["▁O", "ù", "▁sont", "-", "elles", "?"],
        cti_scores=[0.33, 0.03, 0.21, 0.52, 4.49, 0.01],
        cci_scores=[
            CCIOutput(
                cti_idx=4,
                cti_token="elles",
                cti_score=4.49,
                contextual_prefix="Les filles étaient loin. Où sont-elles",
                contextless_prefix="Où sont-ils",
                input_context_scores=[0.0, 0.0, 0.0, 0.0, 0.0],
                output_context_scores=[0.0, 0.01, 0.0, 0.0, 0.0],
            )
        ],
    )
    cli_out = attribute_context(in_out_ctx_encdec_langtag_whitespace_sep)
    assert round_scores(cli_out) == expected_output
