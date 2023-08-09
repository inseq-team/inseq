..
    Copyright 2023 The Inseq Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

#######################################################################################################################
Attributing Distributed LLMs with Petals
#######################################################################################################################

What is Petals?
-------------------------------------

`Petals <https://github.com/bigscience-workshop/petals>`__ is a framework enabling large language models usage without 
the need of high-end GPUs, exploiting the potential of distributed training and inference. With Petals, you can join 
compute resources with other people over the Internet and run large language models such as LLaMA, Guanaco, or BLOOM 
right from your desktop computer or Google Colab. See the `official tutorial <https://colab.research.google.com/drive/1uCphNY7gfAUkdDrTx21dZZwCOUDCMPw8?usp=sharing>`__ and the `paper <https://arxiv.org/pdf/2209.01188.pdf>`__ showcasing 
``petals`` for more details.

.. image:: https://camo.githubusercontent.com/58732a64488a9be928e25f3e60e3692b989ffe212ac86cb4902d8df20a042b03/68747470733a2f2f692e696d6775722e636f6d2f525459463379572e706e67
  :align: center
  :width: 800
  :alt: Visualization of the Tuned Lens approach from Belrose et al. (2023)

Since ``petals`` allows for gradient computations to take place on multiple machines and is mostly compatible with the
Huggingface Transformers library, it can be used alongsides ``inseq`` to attribute large LLMs such as LLaMA 65B or 
Bloom 175B. This tutorial will show how to load a LLM from ``petals`` and use it to attribute a generated sequence.

Attributing LLMs with Petals
-------------------------------------

First, we need to install ``petals`` and ``inseq`` with ``pip install inseq petals``. Then, we can load a LLM from 
``petals`` and attribute it with ``inseq``. For this tutorial, we will use the LLaMA 65B model, which can be loaded as 
follows:

.. code-block:: python

    from petals import AutoDistributedModelForCausalLM

    model_name = "enoch/llama-65b-hf"
    model = AutoDistributedModelForCausalLM.from_pretrained(model_name).cuda()


We can now test a prompt of interest to see whether the model would provide the correct response:

.. code-block:: python

    from transformers import AutoTokenizer

    prompt = (
        "Option 1: Take a 50 minute bus, then a half hour train, and finally a 10 minute bike ride.\n"
        "Option 2: Take a 10 minute bus, then an hour train, and finally a 30 minute bike ride.\n"
        "Which of the options above is faster to get to work?\n"
        "Answer: Option "
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, add_bos_token=False)
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()

    # Only 1 token is generated
    outputs = model.generate(inputs, max_new_tokens=1)
    print(tokenizer.decode(outputs[0]))

    #>>> [...] The answer is Option 1

We can see that the model correctly predicts Option 1 to be the shortest option. Now, we can use ``inseq`` to attribute
the model's prediction to understand which features played a relevant role in determining the model's answer.
Exploiting the advanced features of the ``inseq`` library, we can easily perform a contrastive attribution using
:func:`~inseq.attr.step_functions.contrast_prob_diff_fn` between 1 and 2 as target for gradient attribution (see our
`tutorial <https://github.com/inseq-team/inseq/blob/main/examples/inseq_tutorial.ipynb>`__ for more details).

.. code-block:: python

    out = inseq_model.attribute(
        prompt,
        prompt + "1",
        attributed_fn="contrast_prob_diff",
        contrast_targets=prompt + "2",
        step_scores=["contrast_prob_diff", "probability"],
    )

    # Attributing with input_x_gradient...: 100%|██████████| 80/80 [00:37<00:00, 37.55s/it]

Our attribution results are now stored in the ``out`` variable, and have exactly the same format as the ones obtained
with any other Huggingface decoder-only model. We can now visualize the attribution results using the 
:meth:`~inseq.FeatureAttributionOutput.show` method, specifying the aggregation of our choice. Here we will use the sum
of ``input_x_gradient`` scores across all 8192 dimensions of model input embeddings, without normalizing them to sum to
1:

.. code-block:: python

    out.show(aggregator="sum", normalize=False)

.. raw:: html

    <div class="html-example">
        <iframe frameborder="0" scale="0.75" src="../_static/petals_llama_reasoning_contrastive.htm"></iframe>
    </div>

From the results we can observe that the model is generally upweighting ``minutes`` tokens, while attribution scores
for ``hour`` are less clear-cut. We can also observe that the model predicts Option 1 with a probability of ~53% 
(``probability``), which is roughly 8% higher than the contrastive option 2 (``contrast_prob_diff``). In light of this,
we could formulate the hypothesis that attributions are not very informative because of the relatively low confidence
of the model in its prediction.

.. warning::

    While most methods relying on prediction should work normally with ``petals``, methods requiring access to model
    internals such as ``attention`` are not currently supported.
