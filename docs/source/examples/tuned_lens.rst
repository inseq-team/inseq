..
    Copyright 2023 The Inseq Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

#######################################################################################################################
Estimating Prediction Confidence with Tuned Lens
#######################################################################################################################

The Tuned Lens method
---------------------

.. warning::

    The tutorial is deprecated and won't work with the most recent release of ``tuned-lens``. It will be updated as
    soon as possible.


.. note::

    This tutorial adopts the "Tuned Lens" name for the affine transformation proposed by
    `Belrose et al. (2023) <https://arxiv.org/abs/2303.08112>`__. We note that the Linear Shortcut method proposed by
    `Yom Din et al. (2023) <https://arxiv.org/abs/2303.09435>`__ can be used for the same purpose, training a linear
    transformation instead.

`Belrose et al. (2023) <https://arxiv.org/abs/2303.08112>`__ and
`Yom Din et al. (2023) <https://arxiv.org/abs/2303.09435>`__ introduced a new promising category of approaches to inspect how
predictions are progressively formed across the layers of Transformers-based language models. By training projections
mapping hidden states of intermediate layers to last layer's space, authors overcome the assumption of
a common space shared by all Transformer layers adopted by
`previous work <https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens>`__. Tuned lens
predictions are more aligned to model's predictive distribution and more faithful to the internal feature importance
leveraged by the model.

.. image:: https://d3i71xaburhd42.cloudfront.net/5a1524597b76b67ca8b34fcc6ef8125fd5ce2b3e/2-Figure2-1.png
  :align: center
  :width: 300
  :alt: Visualization of the Tuned Lens approach from Belrose et al. (2023)

An interesting application of the tuned lens leverages the **depth** (i.e. number of model layers) at which a target
token starts dominating the tuned lens' predictive distribution as an indication of **model confidence** and
**example difficulty**. This follows the intuition that simple examples would require less computation, and contrary to
previous work **does not involve further training** of the language model. In this example by Belrose et al (2023),
when a 12B GPT-like model is asked to complete the first sentence of Charles Dickens' "A Tale of
Two Cities", its latent predictions become very confident at early layers, suggesting some degree of memorization.

.. image:: https://d3i71xaburhd42.cloudfront.net/5a1524597b76b67ca8b34fcc6ef8125fd5ce2b3e/16-Figure13-1.png
  :align: center
  :width: 800
  :alt: Visualization of top tuned lens predictions for every layer of the Pythia 12B model at every generation step,
            when asked to complete the first sentence of Charles Dickens' "A Tale of Two Cities".

Adding Tuned Lens Confidence to Inseq
-------------------------------------

Thanks to Inseq extensible design, it is straightforward to integrate the tuned lens method into a new step function
to predict model confidence at every generation step. To do so, we use the
`tuned-lens <https://github.com/AlignmentResearch/tuned-lens>`__ library by Belrose et al.
to extract the tuned lens' predictions at every layer, and then convert the depth at which the model starts predicting
the target token to a confidence score (1 = highest confidence, 0 = lowest confidence, not predicted).

The first step is to install the tuned lens library using ``pip install tuned-lens``, and define our custom step function:

.. code-block:: python

    from tuned_lens.nn.lenses import Lens
    from tuned_lens.residual_stream import record_residual_stream
    import torch

    def confidence_from_prediction_depth(
        # Default arguments for Inseq step functions
        args,
        # Extra arguments for our use case
        lens: Lens,
        # We use kwargs to collect unused default arguments
        **kwargs,
    ):
        """Returns 1 - the ratio of layers after which tuned lens projections over vocabulary
        become aligned with the target token. This can be used as an indication of confidence in
        model prediction. If the token is not predicted by the model, 0% is returned.

        E.g. Using a 12-layer GPT-2 model, and the prompt "Hello ladies and",
        if the target token is "gentlemen" and the tuned lens starts predicting it from layer 8 onwards,
        the returned score is 1 - 8/14 ~= 0.429, indicating good confidence.
        14 is the number of layers in the model, plus the embedding layer, plus 1 to account for the case
        where the token is not predicted by the model.
        """
        batch = attribution_model.formatter.convert_args_to_batch(args)
        # Record activations at every model layer
        with record_residual_stream(attribution_model.model) as stream:
            outputs = attribution_model.get_forward_output(batch, use_embeddings=False)
        # Select last token activations
        stream = stream.map(lambda x: x[..., -1, :])
        # Compute logits for each layer emebedding layer + n_layers
        hidden_lps = stream.zip_map(
            lambda h, i: lens.forward(h, i).log_softmax(dim=-1),
            range(len(stream) - 1),
        )
        # Add last layer's logits
        hidden_lps.layers.append(
            outputs.logits.log_softmax(dim=-1)[..., -1, :]
        )
        num_layers = len(hidden_lps)
        probs = hidden_lps.map(lambda x: x.exp() * 100)
        probs = torch.stack(list(probs))
        top_idx_per_layer = probs.abs().topk(1, dim=-1).indices.squeeze(-1).reshape(-1, num_layers)
        if args.target_ids.ndim == 0:
            args.target_ids = args.target_ids.unsqueeze(0)
        # Set to max denominator to return 0 only if the target token is not predicted by the model
        indices = torch.ones_like(args.target_ids) * (num_layers + 1)
        for i, t in enumerate(args.target_ids):
            pos = torch.where(top_idx_per_layer[i, :] == t.int())[0]
            if pos.numel() > 0:
                indices[i] = pos[0] + 1
        # We add 1 to num_layers to ensure that the score is 0
        # only if the target token is not predicted by the model
        return 1 - (indices / (num_layers + 1))

Now we can simply register the function, load the lens corresponding to the model we want to use, and run the attribution:

.. code-block:: python

    import inseq
    from tuned_lens.nn.lenses import TunedLens

    model = inseq.load_model("gpt2", "input_x_gradient")

    # Load tuned lens for the model from https://hf.co/spaces/AlignmentResearch/tuned-lens
    tuned_lens = TunedLens.load("gpt2", map_location="cpu")

    inseq.register_step_function(
        fn=confidence_from_prediction_depth,
        identifier="confidence",
    )

    out = model.attribute(
        "Hello ladies and",
        lens=tuned_lens,
        device="cpu",
        step_scores=["confidence"],
    )

.. raw:: html

    <div class="html-example">
        <iframe frameborder="0" scale="0.75" src="../_static/tuned_lens.htm"></iframe>
    </div>

We can see that the row ``confidence``, corresponding to the confidence score we defined above, is added at the end of
the attribution matrix, showing high model confidence for function words and multiword expressions endings
(e.g. "Board of Directors", "ladies and gentlemen"). Since we are estimating model confidence on the model's naturally
generated output, all confidence scores will be greater than 0, since this value is reserved for the case where the
target token is not predicted at all.

We can now repeat the experiment while constraining a target generation of our choice:

.. code-block:: python

    out = model.attribute(
        "Hello ladies and",
        # Custom target generation
        "Hello ladies and gentlemen, members of the jury",
        lens=tuned_lens,
        device="cpu",
        step_scores=["confidence"],
    )

.. raw:: html

    <div class="html-example">
        <iframe frameborder="0" scale="0.75" src="../_static/tuned_lens_force.htm"></iframe>
    </div>

We see that some of the forced tokens are assigned a confidence score of 0 in this case.

.. warning::

    The above example aims to show a possible easy integration of ``tuned-lens`` into Inseq, but has a number of limitations.

    - The entire computation using the method above is carried out on CPUs, since device placement is not handled.

    - The tuned lens library currently supports only decoder-only GPT-like models, so the method cannot be used as-is for encoder-decoders like T5 and BART.

    - Tuned lens authors provide a collection of pre-tuned lenses for popular models `here <https://huggingface.co/spaces/AlignmentResearch/tuned-lens/tree/main/lens>`__. If your model of interest is not available, you will need to train a tuned lens for it yourself, which can be done using the `tuned-lens <https://github.com/AlignmentResearch/tuned-lens>`__ codebase.

    - While step functions can generally be also used as attribution targets, the method above does not support this use case in its current form.
