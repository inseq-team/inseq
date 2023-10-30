from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch
from jaxtyping import Float, Float32, Int64
from transformers import PreTrainedModel

TextInput = Union[str, Sequence[str]]


@dataclass
class TokenWithId:
    token: str
    id: int

    def __str__(self):
        return self.token

    def __eq__(self, other: Union[str, int, "TokenWithId"]):
        if isinstance(other, str):
            return self.token == other
        elif isinstance(other, int):
            return self.id == other
        elif isinstance(other, TokenWithId):
            return self.token == other.token and self.id == other.id
        else:
            return False


@dataclass
class TextSequences:
    targets: TextInput
    sources: Optional[TextInput] = None


OneOrMoreIdSequences = Sequence[Sequence[int]]
OneOrMoreTokenSequences = Sequence[Sequence[str]]
OneOrMoreTokenWithIdSequences = Sequence[Sequence[TokenWithId]]
OneOrMoreAttributionSequences = Sequence[Sequence[float]]

IndexSpan = Union[Tuple[int, int], Sequence[Tuple[int, int]]]

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

StepAttributionTensor = Union[GranularStepAttributionTensor, TokenStepAttributionTensor]

# One attribution score per embedding value for every attributed token in attributed_seq
# for all generated tokens in generated_seq. Produced by aggregating GranularStepAttributionTensor
# across multiple steps and separating batches.
GranularSequenceAttributionTensor = Float32[torch.Tensor, "attributed_seq_len generated_seq_len embed_size"]

# One attribution score for every token in attributed_seq for every generated token
# in generated_seq. Produced by aggregating GranularSequenceAttributionTensor over the last dimension,
# or by aggregating TokenStepAttributionTensor across multiple steps and separating batches.
TokenSequenceAttributionTensor = MultipleScoresPerSequenceTensor

SequenceAttributionTensor = Union[GranularSequenceAttributionTensor, TokenSequenceAttributionTensor]

# For Huggingface it's a string identifier e.g. "t5-base", "Helsinki-NLP/opus-mt-en-it"
# For Fairseq it's a tuple of strings containing repo and model name
# e.g. ("pytorch/fairseq", "transformer.wmt14.en-fr")
ModelIdentifier = str  # Union[str, Tuple[str, str]]
ModelClass = PreTrainedModel

AttributionForwardInputs = Union[IdsTensor, EmbeddingsTensor]
AttributionForwardInputsPair = Union[
    Tuple[IdsTensor, IdsTensor],
    Tuple[EmbeddingsTensor, EmbeddingsTensor],
]
OneOrTwoAttributionForwardInputs = Union[AttributionForwardInputs, AttributionForwardInputsPair]
