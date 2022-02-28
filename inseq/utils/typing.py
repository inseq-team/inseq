from typing import Sequence, Tuple, Union

from torch import long
from torchtyping import TensorType


TextInput = Union[str, Sequence[str]]

OneOrMoreIdSequences = Sequence[Sequence[int]]
OneOrMoreTokenSequences = Sequence[Sequence[str]]
OneOrMoreAttributionSequences = Sequence[Sequence[float]]

IdsTensor = TensorType["batch_size", "seq_len", long]
TargetIdsTensor = TensorType["batch_size", long]
EmbeddingsTensor = TensorType["batch_size", "seq_len", "embed_size", float]
MultiStepEmbeddingsTensor = TensorType["batch_size_x_n_steps", "seq_len", "embed_size", float]
VocabularyEmbeddingsTensor = TensorType["vocab_size", "embed_size", float]
FullLogitsTensor = TensorType["batch_size", "seq_len", float]

DeltaOutputTensor = TensorType["batch_size", float]
AttributionOutputTensor = TensorType["batch_size", "seq_len", float]

# For Huggingface it's a string identifier e.g. "t5-base", "Helsinki-NLP/opus-mt-en-it"
# For Fairseq it's a tuple of strings containing repo and model name
# e.g. ("pytorch/fairseq", "transformer.wmt14.en-fr")
ModelIdentifier = Union[str, Tuple[str, str]]

AttributionForwardInputs = Union[IdsTensor, EmbeddingsTensor]
AttributionForwardInputsPair = Union[
    Tuple[IdsTensor, IdsTensor],
    Tuple[EmbeddingsTensor, EmbeddingsTensor],
]
OneOrTwoAttributionForwardInputs = Union[AttributionForwardInputs, AttributionForwardInputsPair]
