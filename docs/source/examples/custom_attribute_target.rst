..
    Copyright 2022 The Inseq Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

#######################################################################################################################
Custom Attribution Targets for Contrastive Attribution
#######################################################################################################################

In this tutorial we will see how to customize the target function used by Inseq to compute attributions, to enable some interesting use
cases of feature attribution methods.

.. note::

    The Inseq library comes with a list of pre-defined step scores functions such as ``probability`` and ``entropy``. By passing one or more
    score names when using ``model.attribute``, these scores will be computed from model outputs and returned in the ``step_scores`` dictionary
    of the output objects. The list of all available scores is available as ``inseq.list_step_functions``, and new scores can be added with
    ``inseq.register_step_function``.


Besides providing useful statistics about model predictive distribution, step score functions are also used as targets when computing feature
attributions. The default behavior of the library is to use next token probability (i.e. the ``probability`` step score) as the attribution target.
This is a fairly standard practice, considering that most studies perform attributions using output logits as targets, and that the softmax
transformation for going from logits to probabilities doesn't affect the attribution scores.

Intuitively, scores produced by attributing next token's probability answer the question "Which elements of the input sequence are
the most relevant to produce the next generation step?". High scores (both positive and negative, depending on the output range
of the attribution method) for a generation step can then be interpreted as input values that heavily impact next token production.

While interesting, this question is not the only one that could be answered by gradient-based methods. For example, we might be interested in
knowing why our model generated its output sequence rather than another one that we consider to be more likely. The paper `"Interpreting Language Models
with Contrastive Explanations" <https://arxiv.org/abs/2202.10419>`__ by Yin and Neubig (2022) suggest that such question can be answered
by complementing the output probabilities with the ones from their contrastive counterpart, and using the difference between the two as attribution
target.

We can define such attribution function using the standard template adopted by Inseq. The :class:`~inseq.attr.step_functions.StepFunctionDecoderOnlyArgs` and :class:`~inseq.attr.step_functions.StepFunctionEncoderDecoderArgs` classes are used for convenience to encapsulate all default arguments passed to step functions, namely:

- :obj:`attribution_model`: the attribution model used to compute attributions.

- :obj:`forward_output`: the output of the forward pass of the attribution model.

- :obj:`target_ids`: the ids corresponding to the next predicted tokens for the current generation step.

- :obj:`ids`, :obj:`embeddings` and :obj:`attention mask` corresponding to the model input at the present step, including inputs for the encoder in case of encoder-decoder models.

.. code-block:: python

    from inseq.attr.step_functions import probability_fn, StepFunctionArgs

    # Simplified implementation of inseq.attr.step_functions.contrast_prob_diff_fn
    # Works only for encoder-decoder models!
    def example_prob_diff_fn(
        # Default arguments for all step functions
        args: StepFunctionArgs,
        # Extra arguments for our use case
        contrast_ids,
        contrast_attention_mask,
    ):
        """Custom attribution function returning the difference between next step probability for
        candidate generation vs. a contrastive alternative, answering the question "Which features
        were salient in deciding to pick the selected token rather than its contrastive alternative?"

        Extra args:
            contrast_ids: Tensor containing the ids of the contrastive input to be compared to the
                regular one.
            contrast_attention_mask: Tensor containing the attention mask for the contrastive input
        """
        # We truncate contrastive ids and their attention map to the current generation step
        device = args.attribution_model.device
        len_inputs = args.decoder_input_ids.shape[1]
        contrast_decoder_input_ids = contrast_ids[:, : len_inputs].to(device)
        contrast_decoder_attention_mask = contrast_attention_mask[:, : len_inputs].to(device)
        # We select the next contrastive token as target
        contrast_target_ids = contrast_ids[:, len_inputs].to(device)
        # Forward pass with the same model used for the main generation, but using contrastive inputs instead
        contrast_output = args.attribution_model.model(
            inputs_embeds=args.encoder_input_embeds,
            attention_mask=args.encoder_attention_mask,
            decoder_input_ids=contrast_decoder_input_ids,
            decoder_attention_mask=contrast_decoder_attention_mask,
        )
        # Return the prob difference as target for attribution
        model_probs = probability_fn(args)
        args.forward_output = contrast_output
        args.target_ids = contrast_target_ids
        contrast_probs = probability_fn(args)
        return model_probs - contrast_probs

Besides common arguments such as the attribution model, its outputs after the forward pass and all the input ids
and attention masks required by |:hugging_face:| Transformers, we provide contrastive ids and their attention mask in input to
compute the difference between original and contrastive probabilities. The output of the function is what is used to
compute the gradients with respect to the input.

Now that we have our custom attribution function, integrating it in Inseq is very easy:

.. code-block:: python

    import inseq

    # Register the function defined above
    # Since outputs are still probabilities, contiguous tokens can still be aggregated using product
    inseq.register_step_function(
        fn=example_prob_diff_fn,
        identifier="example_prob_diff",
        aggregate_map={"span_aggregate": lambda x: x.prod(dim=1, keepdim=True)},
    )

    attribution_model = inseq.load_model("Helsinki-NLP/opus-mt-en-it", "saliency")

    # Pre-compute ids and attention map for the contrastive target
    contrast = attribution_model.encode("Ho salutato la manager", as_targets=True)

    # Perform the contrastive attribution:
    # Regular (forced) target -> "Non posso crederci."
    # Contrastive target      -> "Non posso crederlo."
    # contrast_ids & contrast_attention_mask are kwargs defined in the function definition
    out = attribution_model.attribute(
        "I said hi to the manager",
        "Ho salutato il manager",
        attributed_fn="example_prob_diff",
        contrast_ids=contrast.input_ids,
        contrast_attention_mask=contrast.attention_mask,
        attribute_target=True,
        # We also visualize the step score
        step_scores=["example_prob_diff"]
    )

    # Weight attribution scores by the difference in logits
    out.weight_attributions("example_prob_diff")
    out.show()


.. raw:: html

    <div class="html-example">
        <iframe frameborder="0" scale="0.75" src="../_static/contrastive_example.htm"></iframe>
    </div>

From this example, we see that the masculine Italian determiner "il" is 70% more likely than its feminine counterpart "la" before "manager",
and that the model is mostly influenced by the word manager itself. A textbook example of gender bias in machine translation!
We can also see how the divergence between the two generations has almost no impact on following tokens, if we weight them by the difference in log probabilities.

The contrastive attribution function showcased above is already registered in Inseq under the name ``contrast_prob_diff``, give it a try!

.. note::
    The ``aggregate_map`` argument is useful to inform the library about which functions should be used when aggregating
    step scores (not attributions!) using ``Aggregator`` classes. In this example, we specify that when aggregating over multiple tokens using
    the ``ContiguousSpanAggregator``, we can simply take the product of the computed probability difference as their aggregated score.
