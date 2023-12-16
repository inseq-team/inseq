from inseq.commands.attribute_context import AttributeContextArgs

src_tgt_ctx_s2s_special_tok_sep = AttributeContextArgs(
    model_name_or_path="context-mt/scat-marian-small-target-ctx4-cwd0-en-fr",
    input_context_text="The girls were away.",
    input_current_text="Where are they?",
    output_template="{context} <brk> {current}",
    input_template="{context} <brk> {current}",
)

src_tgt_ctx_s2s_special_tok_sep = AttributeContextArgs(
    model_name_or_path="context-mt/scat-marian-small-ctx4-cwd1-en-fr",
    input_context_text="The girls were away.",
    input_current_text="Where are they?",
    output_template="{current}",
    input_template="{context} <brk> {current}",
)

src_tgt_ctx_s2s_whitespace_sep = AttributeContextArgs(
    model_name_or_path="Helsinki-NLP/opus-mt-en-fr",
    input_context_text="The girls were away.",
    input_current_text="Where are they?",
    output_template="{context} {current}",
    input_template="{context} {current}",
)
