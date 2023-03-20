..
    Copyright 2023 The Inseq Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

#######################################################################################################################
Attributing Multilingual MT Models
#######################################################################################################################

Inseq supports attribution of multilingual MT models such as `mBART <https://arxiv.org/abs/2008.00401>`__,
`M2M-100 <https://dl.acm.org/doi/abs/10.5555/3546258.3546365>`__ and `NLLB <https://arxiv.org/abs/2207.04672>`__.

These models differ from standard encoder-decoder systems in that you will have to specify the source and target
languages, which are used to include a flag in the input to the model. In the following example we attribute a pair
of inputs using M2M-100:

.. code-block:: python

    import inseq
    from inseq.data.aggregator import SubwordAggregator

    model = inseq.load_model(
        "facebook/m2m100_418M",
        "input_x_gradient",
        # The tokenizer_kwargs are used to specify the source and target languages upon initialization
        tokenizer_kwargs={"src_lang": "en", "tgt_lang": "it"},
    )

    out = model.attribute(
        "Did you know? The Inseq library is very flexible!",
        # Step the correct BOS language token
        generation_args={"forced_bos_token_id": model.tokenizer.lang_code_to_id["it"]},
        attribute_target=True,
        step_scores=["probability"],
    )
    # Aggregate the attribution scores at subword level
    out.aggregate().show(aggregator=SubwordAggregator)

.. raw:: html

    <div class="html-example">
        <iframe frameborder="0" scale="0.75" src="/_static/m2m_it_example.htm"></iframe>
    </div>

Probability scores for the target language id should be disregarded, since the token is manually set before generation.
The language ids are model-specific and can be found in the Hugging Face Hub repositories of the models.
