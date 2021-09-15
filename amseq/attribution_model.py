from typing import NoReturn, Tuple, Union

import logging
import warnings

import torch
from captum.attr import (
    IntegratedGradients,
    InterpretableEmbeddingBase,
    configure_interpretable_embedding_layer,
)
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from amseq.outputs import GradientAttributionOutput, TokenizedOutput

logger = logging.getLogger(__name__)


class AttributionModel:
    """Performs  attribution for any seq2seq model in the HF Hub.

    Attributes:
        model (AutoModelForSeq2SeqLM): the seq2seq model on which
            attribution is performed.
        tokenizer (AutoTokenizer): the tokenizer associated to the model.
        device (torch.device): the device on which the model is run (CPU or GPU).
        pad_id (int): the id of the pad token.
        eos_id (int): the id of the end of sequence token.
        bos_id (int): the id of the beginning of sequence token.
        encoder_int_embeds (InterpretableEmbeddingBase): the interpretable embedding
            layer for the encoder.
        decoder_int_embeds (InterpretableEmbeddingBase): the interpretable embedding
            layer for the decoder.
        encoder_embed_scale (float, optional): scale factor for encoder embeddings.
        decoder_embed_scale (float, optional): scale factor for decoder embeddings.
    """

    def __init__(self, model_name_or_path: str, **kwargs) -> NoReturn:
        """
        Initialize the AttributionModel with a Huggingface-compatible seq2seq model.
        Performs the setup for model and embeddings.

        Args:
            model_name_or_path (str): the name of the model in the
                Huggingface Hub or path to folder containing local model files.
            **kwargs: additional arguments to the model and the tokenizer.
        """
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pad_id = self.model.config.pad_token_id
        self.eos_id = self.model.config.eos_token_id
        self.bos_id = self.model.config.decoder_start_token_id
        self.encoder_embed_scale = 1.0
        self.decoder_embed_scale = 1.0
        self.setup_model()
        self.configure_embeddings()

    def setup_model(self) -> NoReturn:
        """Move the model to device and in eval mode.
        For now only greedy decoding (num_beams=1) is supported.
        """
        self.model.to(self.device)
        self.model.eval()
        self.model.zero_grad()
        self.model.config.num_beams = 1

    def configure_embeddings(self) -> NoReturn:
        """Configure embeddings as interpretable embeddings
        Necessary for gradient-attribution methods.
        """
        encoder = self.model.get_encoder()
        decoder = self.model.get_decoder()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if not isinstance(encoder.embed_tokens, InterpretableEmbeddingBase):
                self.encoder_int_embeds = configure_interpretable_embedding_layer(
                    encoder, "embed_tokens"
                )
            if not isinstance(decoder.embed_tokens, InterpretableEmbeddingBase):
                self.decoder_int_embeds = configure_interpretable_embedding_layer(
                    decoder, "embed_tokens"
                )
        if hasattr(encoder, "embed_scale"):
            self.encoder_embed_scale = encoder.embed_scale
        if hasattr(decoder, "embed_scale"):
            self.decoder_embed_scale = decoder.embed_scale

    def tokenize(self, text: str, return_ref: bool = False) -> TokenizedOutput:
        """Tokenize a text, producing a TokenizedOutput

        Args:
            text (str): the text to tokenize.
            return_ref (bool, optional): if True, reference token ids are returned.

        Returns:
            TokenizedOutput: the tokenized text.
        """
        ids = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.tokenizer.max_len_single_sentence,
        )
        input_len = len(ids) - 1  # eos token excluded
        input_ids = torch.tensor([ids], device=self.device, dtype=torch.long)
        attention_mask = torch.ones_like(
            input_ids, dtype=torch.long, device=self.device
        )
        # Reference input is a seq of pad ids matching original length.
        ref_input_ids = (
            None
            if not return_ref
            else torch.tensor(
                [[self.pad_id] * input_len + [self.eos_id]],
                device=self.device,
                dtype=torch.long,
            )
        )
        return TokenizedOutput(input_ids, attention_mask, ref_input_ids)

    def forward_func(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_inputs_embeds: torch.Tensor,
        return_prediction: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        output = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
        )
        logits = output.logits[:, -1, :]
        score = logits.max(-1).values
        next_token = torch.argmax(logits, dim=-1)
        if return_prediction:
            return score, next_token
        return score

    def attribute(
        self, text: str, n_steps: int = 50, batch_size: int = 50
    ) -> GradientAttributionOutput:
        """Perform attribution for a given text.

        Args:
            TODO

        Returns:
            TODO

        Examples:
            .. code:: python

                >>> model = AttributionModel('Helsinki-NLP/opus-mt-en-it')
                >>> attr_out = model.attribute('Hello world!')
        """
        logger.info(f'Original: "{text}"')
        # Build inputs
        out_tok = self.tokenize(text, return_ref=True)
        decoder_input_ids = torch.ones((1, 1), dtype=torch.long, device=self.device)
        decoder_input_ids *= self.bos_id
        input_embeds = self.encoder_int_embeds.indices_to_embeddings(out_tok.input_ids)
        input_embeds = input_embeds * self.encoder_embed_scale
        ref_embeds = self.encoder_int_embeds.indices_to_embeddings(
            out_tok.ref_input_ids
        )
        ref_embeds = ref_embeds * self.encoder_embed_scale
        # Define generator function for tqdm progress bar
        next_token = None

        def gen_next_token():
            while next_token is None or next_token != self.eos_id:
                yield

        # Define the attribution method
        ig = IntegratedGradients(self.forward_func)
        attrs, deltas = None, []
        decoded_string = ""
        pbar = tqdm(gen_next_token())
        for _ in pbar:
            with torch.no_grad():
                if next_token is not None:
                    decoded_string += self.tokenizer.convert_ids_to_tokens(next_token)[
                        0
                    ]
                    pbar.set_description(f"Generating: {decoded_string}")
                    decoder_input_ids = torch.cat(
                        (decoder_input_ids, next_token[:, None]), dim=-1
                    )
                decoder_input_embeds = (
                    self.decoder_int_embeds.indices_to_embeddings(decoder_input_ids)
                    * self.decoder_embed_scale
                )
                _, next_token = self.forward_func(
                    input_embeds,
                    out_tok.attention_mask,
                    decoder_input_embeds,
                    return_prediction=True,
                )
                attr, delta = ig.attribute(
                    input_embeds,
                    baselines=ref_embeds,
                    additional_forward_args=(
                        out_tok.attention_mask,
                        decoder_input_embeds,
                    ),
                    return_convergence_delta=True,
                    n_steps=n_steps,
                    internal_batch_size=batch_size,
                )
                attr = attr.sum(dim=-1).squeeze(0)
                attr = attr / torch.norm(attr)
                deltas.append(float(delta.cpu()[0]))
                if attrs is None:
                    attrs = attr.unsqueeze(1)
                else:
                    attrs = torch.cat([attrs, attr.unsqueeze(1)], dim=1)
        decoder_input_ids = torch.cat((decoder_input_ids, next_token[:, None]), dim=-1)
        tgt_ids = decoder_input_ids.squeeze(0)[1:]
        tgt = self.tokenizer.decode(tgt_ids, skip_special_tokens=True)
        logger.info(f'Generated: "{tgt}"')
        return GradientAttributionOutput(
            self.tokenizer.convert_ids_to_tokens(out_tok.input_ids.cpu().squeeze(0)),
            self.tokenizer.convert_ids_to_tokens(tgt_ids),
            attrs.cpu().numpy(),
            deltas,
        )
