.. inseq documentation entrypoint file

    Copyright 2021 The Inseq Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

##############################
Welcome to Inseq! |:bug:|
##############################

Inseq is a Pytorch-based hackable toolkit to democratize the study of **in**\terpretability for **seq**\uence generation models. At the moment, Inseq supports a wide set of models from the |:hugging_face:| `Transformers <https://github.com/huggingface/transformers/>`__ library and an ever-growing set of feature attribution methods, leveraging in part the widely-used `Captum library <https://captum.ai>`__. For a quick introduction to common use cases, see the :doc:`examples/quickstart` page.

- **Paper:** `https://arxiv.org/abs/2302.13942 <http://arxiv.org/abs/2302.13942>`__
- **Github:** `https://github.com/inseq-team/inseq <https://github.com/inseq-team/inseq>`__
- **PyPI Package:** `https://pypi.org/project/inseq <https://pypi.org/project/inseq>`__
- **MT Gender Bias Demo:** `oskarvanderwal/MT-bias-demo <https://huggingface.co/spaces/oskarvanderwal/MT-bias-demo>`__

Using Inseq, feature attribution maps that can be saved, reloaded, aggregated and visualized either as HTMLs (with Jupyter notebook support) or directly in the console using `rich <https://rich.readthedocs.io/en/latest/>`__. Besides simple attribution, Inseq also supports features like step score extraction, attribution aggregation and attributed functions customization for more advanced use cases. Refer to the guides in the |:bug:| Using Inseq section for more details and examples on specific features.

To give a taste of what Inseq can do in a couple lines of code, here's a snippet doing source-side attribution of an English-to-Italian translation produced by the model ``Helsinki-NLP/opus-mt-en-it`` from |:hugging_face:| Transformers  using the ``IntegratedGradients`` method with 300 integral approximation steps, and returning the attribution convergence delta and token-level prediction probabilties.


.. code-block:: python

    import inseq

    model = inseq.load_model("Helsinki-NLP/opus-mt-en-fr", "integrated_gradients")
    out = model.attribute(
        "The developer argued with the designer because she did not like the design.",
        n_steps=300,
        return_convergence_delta=True,
        step_scores=["probability"],
    )
    out.show()

.. raw:: html

    <div class="html-example">
        <iframe frameborder="0" scale="0.75" src="_static/winomt_example.htm"></iframe>
    </div>

Inseq is still in early development and is currently maintained by a small team of grad students working on interpretability for NLP/NLG led by `Gabriele Sarti <https://gsarti.com>`__. We are working hard to add more features and models. If you have any suggestions or feedback, please open an issue on our `GitHub repository <https://github.com/inseq-team/inseq/issues>`__. Happy hacking! |:bug:|

---

.. toctree::
    :maxdepth: 2
    :caption: Using Inseq 🐛

    examples/quickstart
    examples/pair_comparison
    examples/custom_attribute_target
    examples/attribute_mmt
    examples/locate_gpt2_knowledge
    examples/tuned_lens
    examples/faq

.. toctree::
    :maxdepth: 3
    :caption: API Documentation

    main_classes/main_functions
    main_classes/models
    main_classes/data_classes
    main_classes/feature_attribution
    main_classes/step_functions
