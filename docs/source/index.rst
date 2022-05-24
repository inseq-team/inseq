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

Here is an example of using Inseq to attribute an English-to-Italian translation of the model ``Helsinki-NLP/opus-mt-en-it`` from the
|:hugging_face:| `Transformers <https://github.com/huggingface/transformers/>`__ using the ``IntegratedGradients`` method from the
`Captum library <https://captum.ai>`__.


.. code-block:: python

    import inseq

    model = inseq.load_model("Helsinki-NLP/opus-mt-en-fr", "integrated_gradients")
    out = model.attribute(
        "The developer argued with the designer because she did not like the design.",
        n_steps=300,
        return_convergence_delta=True,
        output_step_probabilities=True,
        internal_batch_size=100,
        include_eos_baseline=False,
    )
    out.show()

.. raw:: html

    <div class="html-example">
        <iframe frameborder="0" scale="0.75" src="/_static/winomt_example.htm"></iframe>
    </div>


.. toctree::
    :maxdepth: 2
    :caption: Using 🐛 Inseq

    examples/pair_comparison
    examples/quickstart

.. toctree::
    :maxdepth: 4
    :caption: Main Classes

    main_classes/models
    main_classes/data_classes
    main_classes/feature_attribution
