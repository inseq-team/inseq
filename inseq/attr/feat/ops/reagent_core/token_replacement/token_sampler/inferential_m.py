import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
from typing_extensions import override

from .base import TokenSampler


class InferentialMTokenSampler(TokenSampler):
    """Sample tokens from a seq-2-seq model

    """

    @override
    def __init__(self, source_tokenizer: AutoTokenizer, sampler_tokenizer: AutoTokenizer, sampler_model: AutoModelWithLMHead) -> None:
        """Constructor

        Args:
            source_tokenizer: A Huggingface AutoTokenizer for decoding the inputs.
            sampler_tokenizer: A Huggingface AutoTokenizer for inference the output.
            sampler_model: A Huggingface AutoModelWithLMHead for inference the output.

        """
        super().__init__()

        self.source_tokenizer = source_tokenizer
        self.sampler_tokenizer = sampler_tokenizer
        self.sampler_model = sampler_model

    @override
    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        """Sample a tensor

        Args:
            inputs: input tensor [batch, sequence]
        
        Returns:
            token_inferences: sampled (placement) tokens by inference

        """
        super().sample(inputs)

        batch_li = []
        for seq_i in torch.arange(inputs.shape[0]):
            seq_li = []
            for pos_i in torch.arange(inputs.shape[1]):

                # first token
                if pos_i == 0:
                   seq_li.append(inputs[seq_i, 0])
                   continue

                # following tokens

                probe_prefix = torch.tensor([self.sampler_tokenizer.encode(self.source_tokenizer.decode(inputs[seq_i, :pos_i]))], device=inputs.device)
                probe_prefix = probe_prefix[:,:-1]  # trim <eos>
                output_replacing_m = self.sampler_model(probe_prefix)
                logits_replacing_m = output_replacing_m['logits']
                logits_replacing_m_last = logits_replacing_m[:,-1]
                id_infer_m = torch.argmax(logits_replacing_m_last, dim=-1)

                seq_li.append(id_infer_m.item())

            batch_li.append(seq_li)
        
        res = torch.tensor(batch_li, device=inputs.device)

        return res

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cpu"

    source_tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir="cache")
    source_model = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir="cache").to(device)
    source_model.eval()

    sampler_tokenizer = AutoTokenizer.from_pretrained("roberta-base", cache_dir="cache")
    sampler_model = AutoModelForCausalLM.from_pretrained("roberta-base", cache_dir="cache").to(device)
    sampler_model.eval()

    sampler = InferentialMTokenSampler(source_tokenizer, sampler_tokenizer, sampler_model)

    text = "This is a test sequence"
    inputs = torch.tensor([ source_tokenizer.encode(text) ], device=device)

    outputs = sampler.sample(inputs)

    print(outputs)
    print(source_tokenizer.decode(outputs[0]))


