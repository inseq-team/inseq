
import math
from typing import Union

import torch
from .base import BaseRationalizer
from .importance_score_evaluator.base import BaseImportanceScoreEvaluator

from typing_extensions import override


class AggregateRationalizer(BaseRationalizer):
    """AggregateRationalizer
    
    """

    def __init__(self, importance_score_evaluator: BaseImportanceScoreEvaluator, batch_size: int, overlap_threshold: int, overlap_strict_pos: bool = True, top_n: float = 0, top_n_ratio: float = 0) -> None:
        """Constructor

        Args:
            importance_score_evaluator: A ImportanceScoreEvaluator
            batch_size: Batch size for aggregate
            overlap_threshold: Overlap threshold of rational tokens within a batch
            overlap_strict_pos: Whether overlap strict to position ot not
            top_n: Rational size
            top_n_ratio: Use ratio of sequence to define rational size

        """
        super().__init__(importance_score_evaluator)

        self.batch_size = batch_size
        self.overlap_threshold = overlap_threshold
        self.overlap_strict_pos = overlap_strict_pos
        self.top_n = top_n
        self.top_n_ratio = top_n_ratio

        assert overlap_strict_pos == True, "overlap_strict_pos = False not been supported yet"

    def get_separate_rational(self, input_ids, tokenizer) -> Union[torch.Tensor, list[list[str]]]:

        tokens = [ [ tokenizer.decode([input_ids[0, i]]) for i in s] for s in self.pos_top_n ]

        return self.pos_top_n, tokens

    @torch.no_grad()
    def rationalize(self, input_ids: torch.Tensor, target_id: torch.Tensor) -> torch.Tensor:
        """Compute rational of a sequence on a target

        Args:
            input_ids: The sequence [batch, sequence] (first dimension need to be 1)
            target_id: The target [batch]

        Return:
            pos_top_n: rational position in the sequence [batch, rational_size]

        """
        assert input_ids.shape[0] == 1, "the first dimension of input (batch_size) need to be 1"

        batch_input_ids = input_ids.repeat(self.batch_size, 1)

        batch_importance_score = self.importance_score_evaluator.evaluate(batch_input_ids, target_id)
        
        important_score_masked = batch_importance_score * torch.unsqueeze(self.importance_score_evaluator.stop_mask, -1)
        self.mean_important_score = torch.sum(important_score_masked, dim=0) / torch.sum(self.importance_score_evaluator.stop_mask)

        pos_sorted = torch.argsort(batch_importance_score, dim=-1, descending=True)

        top_n = self.top_n

        if top_n == 0:
            top_n = int(math.ceil(self.top_n_ratio * input_ids.shape[-1]))

        pos_top_n = pos_sorted[:, :top_n]
        self.pos_top_n = pos_top_n

        if self.overlap_strict_pos:
            count_overlap = torch.bincount(pos_top_n.flatten(), minlength=input_ids.shape[1])
            pos_top_n_overlap = torch.unsqueeze(torch.nonzero(count_overlap >= self.overlap_threshold, as_tuple=True)[0], 0)
            return pos_top_n_overlap
        else:
            token_id_top_n = input_ids[0, pos_top_n]
            count_overlap = torch.bincount(token_id_top_n.flatten(), minlength=input_ids.shape[1])
            token_id_top_n_overlap = torch.unsqueeze(torch.nonzero(count_overlap >= self.overlap_threshold, as_tuple=True)[0], 0)
            # TODO: Convert back to pos
            raise NotImplementedError("TODO")
            

    @override
    def trace_start(self) -> None:
        """Start tracing
        
        """
        super().trace_start()

        self.importance_score_evaluator.trace_start()

    @override
    def trace_stop(self) -> None:
        """Stop tracing
        
        """
        super().trace_stop()

        self.importance_score_evaluator.trace_stop()


@torch.no_grad()
def main():

    from stopping_condition_evaluator.top_k import \
        TopKStoppingConditionEvaluator
    from token_replacement.token_replacer.uniform import UniformTokenReplacer
    from token_replacement.token_sampler.inferential import \
        InferentialTokenSampler
    from token_replacement.token_sampler.postag import POSTagTokenSampler
    from token_replacement.token_sampler.uniform import UniformTokenSampler
    from transformers import AutoModelWithLMHead, AutoTokenizer

    from rationalization.rationalizer.importance_score_evaluator.delta_prob import \
        DeltaProbImportanceScoreEvaluator
    from utils.serializing import serialize_rational

    # ======== model loading ========
    # Load model from Hugging Face
    model = AutoModelWithLMHead.from_pretrained("gpt2-medium")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")

    model.cuda()
    model.eval()
    
    # ======== prepare data ========

    # batch with size 1
    input_string = [
        # "I love eating breakfast in the",
        "When my flight landed in Thailand. I was staying in the capital city of"
        # "When my flight landed in Thailand, I converted my currency and slowly fell asleep. I was staying in the capital city of"
        # "When my flight landed in Thailand, I converted my currency and slowly fell asleep. (I had a terrifying dream about my grandmother, but that's a story for another time). I was staying in the capital city of"
    ]

    # generate prediction 
    input_ids = tokenizer(input_string, return_tensors='pt')['input_ids'].to(model.device)
    generated_input = model.generate(input_ids=input_ids, max_length=80, do_sample=False) 
    print(' generated input -->', [ [ tokenizer.decode(token) for token in seq] for seq in generated_input ])

    # extract target from prediction
    target_id = generated_input[:, input_ids.shape[1]]
    print(' target -->', [ tokenizer.decode(token) for token in target_id ])

    # ======== hyper-parameters ========

    # replacing ratio during importance score updating
    updating_replacing_ratio = 0.3
    # keep top n word based on importance score for both stop condition evaluation and rationalization
    rationale_size_ratio = None
    rational_size = 5
    # stop when target exist in top k predictions
    stop_condition_tolerance = 5

    # Batch size for aggregate
    aggregate_batch_size = 5
    # Overlap threshold of rational tokens within a batch
    overlap_threshold = 3
    # Whether overlap strict to position ot not
    overlap_strict_pos = True

    # ======== rationalization ========
    
    approach_sample_replacing_token = "uniform"
    # approach_sample_replacing_token = "inference"
    # approach_sample_replacing_token = "postag"

    # prepare rationalizer
    if approach_sample_replacing_token == "uniform":
        # Approach 1: sample replacing token from uniform distribution
        rationalizer = AggregateRationalizer(
            importance_score_evaluator=DeltaProbImportanceScoreEvaluator(
                model=model, 
                tokenizer=tokenizer, 
                token_replacer=UniformTokenReplacer(
                    token_sampler=UniformTokenSampler(tokenizer), 
                    ratio=updating_replacing_ratio
                ),
                stopping_condition_evaluator=TopKStoppingConditionEvaluator(
                    model=model, 
                    token_sampler=UniformTokenSampler(tokenizer), 
                    top_k=stop_condition_tolerance, 
                    top_n=rational_size, 
                    top_n_ratio=rationale_size_ratio, 
                    tokenizer=tokenizer
                )
            ), 
            batch_size=aggregate_batch_size,
            overlap_threshold=overlap_threshold,
            overlap_strict_pos=overlap_strict_pos,
            top_n=rational_size, 
            top_n_ratio=rationale_size_ratio
        )
    elif approach_sample_replacing_token == "inference":
        # Approach 2: sample replacing token from model inference
        rationalizer = AggregateRationalizer(
            importance_score_evaluator=DeltaProbImportanceScoreEvaluator(
                model=model, 
                tokenizer=tokenizer, 
                token_replacer=UniformTokenReplacer(
                    token_sampler=InferentialTokenSampler(tokenizer=tokenizer, model=model), 
                    ratio=updating_replacing_ratio
                ),
                stopping_condition_evaluator=TopKStoppingConditionEvaluator(
                    model=model, 
                    token_sampler=InferentialTokenSampler(tokenizer=tokenizer, model=model), 
                    top_k=stop_condition_tolerance, 
                    top_n=rational_size, 
                    top_n_ratio=rationale_size_ratio, 
                    tokenizer=tokenizer
                )
            ), 
            batch_size=aggregate_batch_size,
            overlap_threshold=overlap_threshold,
            overlap_strict_pos=overlap_strict_pos,
            top_n=rational_size, 
            top_n_ratio=rationale_size_ratio
        )
    elif approach_sample_replacing_token == "postag":
        # Approach 3: sample replacing token from uniform distribution on a set of words with the same POS tag
        ts = POSTagTokenSampler(tokenizer=tokenizer, device=input_ids.device) # Initialize POSTagTokenSampler takes time so share it
        rationalizer = AggregateRationalizer(
            importance_score_evaluator=DeltaProbImportanceScoreEvaluator(
                model=model, 
                tokenizer=tokenizer, 
                token_replacer=UniformTokenReplacer(
                    token_sampler=ts, 
                    ratio=updating_replacing_ratio
                ),
                stopping_condition_evaluator=TopKStoppingConditionEvaluator(
                    model=model, 
                    token_sampler=ts, 
                    top_k=stop_condition_tolerance, 
                    top_n=rational_size, 
                    top_n_ratio=rationale_size_ratio, 
                    tokenizer=tokenizer
                )
            ), 
            batch_size=aggregate_batch_size,
            overlap_threshold=overlap_threshold,
            overlap_strict_pos=overlap_strict_pos,
            top_n=rational_size, 
            top_n_ratio=rationale_size_ratio
        )
    else:
        raise ValueError("Invalid approach_sample_replacing_token")
    
    rationalizer.trace_start()

    # rationalization
    pos_rational = rationalizer.rationalize(input_ids, generated_input[:, input_ids.shape[1]])

    # convert results

    print()
    print(f"========================")
    print()
    print(f'Input --> {input_string[0]}')
    print(f'Target --> {tokenizer.decode(target_id[0])}')
    print(f"Rational positions --> {pos_rational}")
    print(f"Rational words -->")
    for i in range(pos_rational.shape[0]):
        ids_rational = input_ids[0, pos_rational[i]]
        text_rational = [ tokenizer.decode([id_rational]) for id_rational in ids_rational ]
        print(f"{text_rational}")

    # output

    serialize_rational(
        "rationalization_results/demo.json", 
        -1, 
        input_ids[0], 
        target_id[0], 
        pos_rational[0], 
        tokenizer, 
        rationalizer.importance_score_evaluator.important_score[0],
        compact=False,
        comments= {
            "message": "This is a demo output. [comments] is an optional field",
            "model": "gpt2-medium",
            "approach_type": approach_sample_replacing_token
        },
        trace_rationalizer=rationalizer
    )

    rationalizer.trace_stop()

if __name__ == '__main__':
    main()
