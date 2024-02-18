import nltk
import torch
from transformers import AutoTokenizer
from typing_extensions import override

from .base import TokenSampler


class POSTagTokenSampler(TokenSampler):
    """Sample tokens from Uniform distribution on a set of words with the same POS tag

    """

    @override
    def __init__(self, tokenizer: AutoTokenizer, device=None) -> None:
        """Constructor

        Args:
            tokenizer: A Huggingface AutoTokenizer.

        """
        super().__init__()

        self.tokenizer = tokenizer

        # extract mapping from postag to words
        # debug_mapping_postag_to_group_word = {}
        mapping_postag_to_group_token_id = {}
        
        for i in range(tokenizer.vocab_size):
            word = tokenizer.decode([i])
            _, tag = nltk.pos_tag([word.strip()])[0]
            if tag not in mapping_postag_to_group_token_id:
                # debug_mapping_postag_to_group_word[tag] = []
                mapping_postag_to_group_token_id[tag] = []
            # debug_mapping_postag_to_group_word[tag].append(word)
            mapping_postag_to_group_token_id[tag].append(i)

            if i % 5000 == 0:
                print(f"[POSTagTokenSampler] Loading vocab from tokenizer - {i / tokenizer.vocab_size * 100:.2f}%")

        # create tag_id for postags
        self.list_postag = [ tag for tag in mapping_postag_to_group_token_id.keys() ]
        num_postags = len(self.list_postag)

        # build mapping from tag_id to word group
        list_group_token_id = [ torch.tensor(mapping_postag_to_group_token_id[postag], dtype=torch.long, device=device) for postag in self.list_postag ]

        # build mapping from token_id to tag_id
        self.mapping_token_id_to_tag_id = torch.zeros([tokenizer.vocab_size], dtype=torch.long, device=device)
        for tag_id, group_token_id in enumerate(list_group_token_id):
            self.mapping_token_id_to_tag_id[group_token_id] = tag_id

        # build mapping from tag_id to token_id
        # postag groups are concat together, index them via compact_idx = group_offsets[tag_id] + group_idx
        self.group_sizes = torch.tensor([ group_token_id.shape[0] for group_token_id in list_group_token_id ], dtype=torch.long, device=device)
        self.group_offsets = torch.sum(torch.tril(torch.ones([num_postags, num_postags], device=device), diagonal=-1) * self.group_sizes, dim=-1)
        self.compact_group_token_id = torch.cat(list_group_token_id)

    @override
    def sample(self, input: torch.Tensor) -> torch.Tensor:
        """Sample a input

        Args:
            input: input tensor [batch, sequence]
        
        Returns:
            token_sampled: A sampled tensor where its shape is the same with the input

        """
        super().sample(input)

        tag_id_input = self.mapping_token_id_to_tag_id[input]
        sample_uniform = torch.rand(input.shape, device=input.device)
        compact_group_idx = (sample_uniform * self.group_sizes[tag_id_input] + self.group_offsets[tag_id_input]).type(torch.long)
        token_sampled = self.compact_group_token_id[compact_group_idx]

        return token_sampled
