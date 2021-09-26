.. inseq documentation entrypoint file

    Copyright 2021 Gabriele Sarti. All rights reserved.

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

    model = inseq.load("Helsinki-NLP/opus-mt-en-it", "integrated_gradients")
    text = "Hello world, today is a good day!"
    out = model.attribute(txt)

.. raw:: html

    <img alt="IntegratedGradients Example Attribution" src="_static/heatmap_helloworld_enit.png" style="max-width: 600px;">