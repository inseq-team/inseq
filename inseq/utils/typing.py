from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from captum.attr._utils.attribution import Attribution
from jaxtyping import Float, Float32, Int64
from transformers import PreTrainedModel

TextInput = str | Sequence[str]

if TYPE_CHECKING:
    from inseq.models import AttributionModel


@dataclass
class TokenWithId:
    token: str
    id: int

    def __str__(self):
        return self.token

    def __eq__(self, other: "str | int | TokenWithId"):
        if isinstance(other, str):
            return self.token == other
        elif isinstance(other, int):
            return self.id == other
        elif isinstance(other, TokenWithId):
            return self.token == other.token and self.id == other.id
        else:
            return False


class InseqAttribution(Attribution):
    """A wrapper class for the Captum library's Attribution class to type hint the ``forward_func`` attribute
    as an :class:`~inseq.models.AttributionModel`.
    """

    def __init__(self, forward_func: "AttributionModel") -> None:
        r"""
        Args:
            forward_func (:class:`~inseq.models.AttributionModel`): The model hooker to the attribution method.
        """
        self.forward_func = forward_func

    attribute: Callable

    @property
    def multiplies_by_inputs(self):
        return False

    def has_convergence_delta(self) -> bool:
        return False

    compute_convergence_delta: Callable

    @classmethod
    def get_name(cls: type["InseqAttribution"]) -> str:
        return "".join([char if char.islower() or idx == 0 else " " + char for idx, char in enumerate(cls.__name__)])


@dataclass
class TextSequences:
    targets: TextInput
    sources: TextInput | None = None


OneOrMoreIdSequences = Sequence[Sequence[int]]
OneOrMoreTokenSequences = Sequence[Sequence[str]]
OneOrMoreTokenWithIdSequences = Sequence[Sequence[TokenWithId]]
OneOrMoreAttributionSequences = Sequence[Sequence[float]]

ScorePrecision = Literal["float32", "float16", "float8"]

IndexSpan = tuple[int, int] | Sequence[tuple[int, int]]
OneOrMoreIndices = int | list[int] | tuple[int, int]
OneOrMoreIndicesDict = dict[int, OneOrMoreIndices]

IdsTensor = Int64[torch.Tensor, "batch_size seq_len"]
TargetIdsTensor = Int64[torch.Tensor, "batch_size"]
ExpandedTargetIdsTensor = Int64[torch.Tensor, "batch_size 1"]
EmbeddingsTensor = Float[torch.Tensor, "batch_size seq_len embed_size"]
MultiStepEmbeddingsTensor = Float[Float, "batch_size_x_n_steps seq_len embed_size"]
VocabularyEmbeddingsTensor = Float[torch.Tensor, "vocab_size embed_size"]
LogitsTensor = Float[torch.Tensor, "batch_size vocab_size"]
ScoreTensor = Float[torch.Tensor, "batch_size other_dims"]
MultiUnitScoreTensor = Float[torch.Tensor, "batch_size n_units other_dims"]
MultiLayerScoreTensor = Float[torch.Tensor, "batch_size n_layers other_dims"]
MultiLayerMultiUnitScoreTensor = Float[torch.Tensor, "batch_size n_layers n_units seq_len seq_len"]
MultiLayerEmbeddingsTensor = Float[torch.Tensor, "batch_size n_layers seq_len embed_size"]

# Step and sequence objects used for stepwise scores (e.g. convergence deltas, probabilities)
SingleScorePerStepTensor = Float32[torch.Tensor, "batch_size"]
SingleScoresPerSequenceTensor = Float32[torch.Tensor, "generated_seq_len"]

# Step and sequence objects used for sequence scores (e.g. attributions over tokens)
MultipleScoresPerStepTensor = Float32[torch.Tensor, "batch_size attributed_seq_len"]
MultipleScoresPerSequenceTensor = Float32[torch.Tensor, "attributed_seq_len generated_seq_len"]

# One attribution score per embedding value for every attributed token
# in a single attribution step. Produced by gradient attribution methods.
GranularStepAttributionTensor = EmbeddingsTensor

# One attribution score per token for a single attribution step
# Either product of aggregation of GranularStepAttributionTensor over last dimension,
# or produced by methods that work at token-level (e.g. attention)
TokenStepAttributionTensor = MultipleScoresPerStepTensor

StepAttributionTensor = GranularStepAttributionTensor | TokenStepAttributionTensor

# One attribution score per embedding value for every attributed token in attributed_seq
# for all generated tokens in generated_seq. Produced by aggregating GranularStepAttributionTensor
# across multiple steps and separating batches.
GranularSequenceAttributionTensor = Float32[torch.Tensor, "attributed_seq_len generated_seq_len embed_size"]

# One attribution score for every token in attributed_seq for every generated token
# in generated_seq. Produced by aggregating GranularSequenceAttributionTensor over the last dimension,
# or by aggregating TokenStepAttributionTensor across multiple steps and separating batches.
TokenSequenceAttributionTensor = MultipleScoresPerSequenceTensor

SequenceAttributionTensor = GranularSequenceAttributionTensor | TokenSequenceAttributionTensor

# For Huggingface it's a string identifier e.g. "t5-base", "Helsinki-NLP/opus-mt-en-it"
# For Fairseq it's a tuple of strings containing repo and model name
# e.g. ("pytorch/fairseq", "transformer.wmt14.en-fr")
ModelIdentifier = str  # Union[str, Tuple[str, str]]
ModelClass = PreTrainedModel

AttributionForwardInputs = IdsTensor | EmbeddingsTensor
AttributionForwardInputsPair = tuple[IdsTensor, IdsTensor] | tuple[EmbeddingsTensor, EmbeddingsTensor]
OneOrTwoAttributionForwardInputs = AttributionForwardInputs | AttributionForwardInputsPair
