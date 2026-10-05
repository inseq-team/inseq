# Grad-ELLM method card (draft for native Inseq documentation)

Grad-ELLM is a target-dependent, token-level attribution method for decoder-only
Llama and Mistral models. It uses internal Q/K/V projections and gradients of a
chosen scalar prediction objective. Use `attributed_fn="logit"` to reproduce the
supplied algorithm; Inseq's usual default objective is probability.

For each selected layer and one next-token prediction, let q be the final valid
prefix query, k_i and v_i the expanded key/value projections at prefix position i,
and g the gradient at the final prefix position of the self-attention output
after o_proj. Q/K are raw linear projections BEFORE RoPE.

With the default `normalize=True`, compute cosine(q, k_i) and min-max normalize
these similarities over valid prefix positions. Constant similarities produce
zero scores. With `normalize=False`, use softmax(q dot k_i / sqrt(query_width))
over valid prefix positions. A layer score is
`relu(sum_hidden(g * v_i * similarity_i))`; scores are summed over the last
`n_layers` layers, without averaging. Padding has zero contribution.

Defaults: n_layers=1, normalize=True, kv_expansion="legacy_repeat",
gradient_source="attention_output". Legacy expansion repeats the entire flattened
K/V vector by the dynamically inferred GQA ratio, matching the supplied code's
ordering. Ordinary MHA has ratio 1 and performs no replication. `grouped`
head replication and `pre_o` gradients are explicit algorithm variants, not the
paper-reproduction defaults.

The method does not produce embedding-component or head-specific attributions:
raw step scores have shape [batch, prefix_tokens], and decoder-only sequence
scores are stored in target_attributions with shape [prefix_and_generated_tokens,
attributed_tokens]. source_attributions is None. Future-token cells are NaN by
Inseq convention. A method-specific identity dimension aggregator preserves both
token axes and reuses Inseq's normal L1 score normalization for show()/aggregate().
That postprocessing is independent of the method's Q/K normalize option.

Each attributed token requires a fresh complete-prefix forward/backward, with
use_cache=False. Hooks are temporary and removed on failure; parameter .grad is
not accumulated. Models must be in eval mode; attribution is incompatible with
torch.inference_mode(). Standard floating-point weights on one CPU or CUDA device
are supported by this prototype. Training, quantization, sharding/offload, other
model families and multi-forward contrastive objectives require separate adapters.
Half/bfloat16 model projections and gradients are promoted to float32 during score
calculation; model forward/backward retains its dtype.

Exact ID replay is an auxiliary internal helper, not a new Inseq-wide public API
proposal. It accepts one unpadded prompt+continuation sequence and retains BOS/EOS.
The indexed-CUDA context bridges the existing Inseq 0.6/0.7 backend-name validation.
Maintainers may replace these helpers with framework-level device/token-ID support.

Validation: see the external package's VALIDATION.md and AUDIT_0_1_5.md. Current
checks use tiny randomly initialized HF models and CPU-only dependencies. Real
checkpoint/CUDA evidence and the target upstream branch's full CI are still needed.

Before an upstream PR: add the verified paper citation/BibTeX and contributor
license information. No author list, DOI or publication identifier is guessed here.
