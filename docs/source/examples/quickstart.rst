.. Quickstart to public methods and common use-cases of the Inseq library

    Copyright 2021 The Inseq Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

###################################
Getting started with Inseq
###################################

The Inseq library is a Pytorch-based toolkit inteded to democratize the access to some common use-cases in the study of sequence generation models for interpretability purposes. At the moment, the library supports the following set of models and techniques:

**Models**

- All the models made available through the `AutoModelForSeq2SeqLM <https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSeq2SeqLM>`_ interface of the |:hugging_face:| `transformers <https://github.com/huggingface/transformers>`_ library (among others, `T5 <https://huggingface.co/docs/transformers/model_doc/t5>`_, `Bart <https://huggingface.co/docs/transformers/model_doc/bart>`_ and all >1000 `MarianNMT <https://huggingface.co/docs/transformers/model_doc/marian>`_ variants) can be used in combination with feature attribution methods.

- All the models made available through the `AutoModelForCausalLM <https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM>`_ interface of the |:hugging_face:| `transformers <https://github.com/huggingface/transformers>`_ library (among others, `GPT-2 <https://huggingface.co/docs/transformers/model_doc/gpt2>`_, `GPT-NeoX <https://huggingface.co/docs/transformers/model_doc/gpt_neox>`_, `Bloom <https://huggingface.co/docs/transformers/model_doc/bloom>`_ and `OPT/Galactica <https://huggingface.co/docs/transformers/model_doc/opt>`__).

**Interpretability Methods**

- At the moment, only gradient-based feature attribution methods sourced from the `Captum <https://captum.ai>`_ library are available, but other popular occlusion and attention-based techniques will soon follow. The list of all available methods can be obtained by using the :meth:`~inseq.list_feature_attribution_methods` method. Each method either points to its original implementation, and is thoroughly documented in its docstring.

Installing Inseq
===================================

The latest version of Inseq can be installed from PyPI using ``pip install inseq``. To gain access to some Inseq functionalities, you will need to install optional dependencies (e.g. use ``pip install inseq[datasets]`` to enable datasets attribution via the Inseq CLI). For installing the dev version and contributing, please follow the instructions in Inseq readme file.

The ``AttributionModel`` class
===================================

The :class:`~inseq.models.AttributionModel` class is a ``torch.nn.Module`` intended to seamlessly wrap any sequence generation Pytorch model to enable its interpretability. More specifically, the class adds the following capabilities to the wrapped model:

- A :meth:`~inseq.models.AttributionModel.load` method to load the weights of the wrapped model from a saved checkpoint, locally or remotely. This is called when using the :meth:`~inseq.load_model` function, which is the suggested way to load a model.

- An :meth:`~inseq.models.AttributionModel.attribute` method used to perform feature attribution using the loaded model.

- Multiple utility methods like :meth:`~inseq.models.AttributionModel.encode` and :meth:`~inseq.models.AttributionModel.embed` that are also used internally by the ``attribute`` method.

``AttributionModel`` children classes belong to two categories: **architectural classes** like :class:`~inseq.models.EncoderDecoderAttributionModel` defines methods that are specific to a certain model architecture, while **framework classes** like :class:`~inseq.models.HuggingfaceModel` specify methods that are specific to a certain modeling framework (e.g. encoding with a tokenizer in |:hugging_face:| transformers). The final class that will be instantiated by the user is a combination of the two, e.g. :class:`~inseq.models.HuggingfaceEncoderDecoderModel` for a sequence-to-sequence model from the |:hugging_face:| transformers library.

When a model is loaded with :meth:`~inseq.load_model`, a :class:`~inseq.attr.feat.FeatureAttribution` can be attached to it to specify which feature attribution technique should be used on it. Different families of attribution methods such as :class:`~inseq.attr.feats.GradientAttribution` are made available, each containing multiple methods (e.g. :class:`~inseq.attr.feats.IntegratedGradientsAttribution`, :class:`~inseq.attr.feats.DeepLiftAttribution`).

The following image provides a visual hierarchy of the division between ``AttributionModel`` and ``FeatureAttribution`` subclasses:

.. image:: ../images/classes.png
  :width: 400
  :alt: Classes diagram for attribution models and feature attribution methods.

The ``attribute`` method
===================================

The :meth:`~inseq.AttributionModel.attribute` method provides a easy to use and flexible interface to generate feature attributions with sequence generation models. In its most simple form, the selected model is used to generate one or more output sequences with default parameters, and then those are attributed with the specified feature attribution method.

.. code-block:: python

    import inseq

    model = inseq.load_model("Helsinki-NLP/opus-mt-en-fr", "saliency")
    out = model.attribute(input_texts="Hello world, here's the Inseq library!")

The ``attribute`` method supports a wide range of customizations. Among others:

- Specifying one string in ``generated_texts`` for every sentence in ``input_texts`` allows attributing custom generation outputs. Useful to answer the question "How would the following output be justified in light of the inputs by the model?".

- ``attr_pos_start`` and ``attr_pos_end`` can be used to attribute only specific spans of the generated output, making the attribution process more efficient when one is only interested in attributions at a specific output step.

- ``output_step_attributions`` will fill the ``step_attributions`` property in the output object with step-by-step attributions that are normally produced but then discarded after converting them in sequence attributions specific to every sequence in the attributed batch.

- ``attribute_target`` can be used to specify that target-side prefix should also be attributed for encoder-decoder models besides the original source-to-target attribution. This would populate the ``target_attribution`` filed in the output, which would otherwise be left empty. In the decoder-only case, the parameter is not used since only the prefix is attributed by default.

- ``step_scores`` allows for computing custom scores at every generation step, with some such as token ``probability`` and output distribution ``entropy`` being defined by default in Inseq.

- ``attributed_fn`` allows defining a custom output function for the model, enabling advanced use cases such as `contrastive explanations <https://arxiv.org/abs/2202.10419>`__.


The ``FeatureAttributionOutput`` class
=======================================

In the code above, the ``out`` object is a :class:`~inseq.FeatureAttributionOutput` instance, containing a list of ``sequence_attributions`` and additional useful ``ìnfo`` regarding the attribution that was performed. In this example ``sequence_attributions`` has length 1 since a single sequence was attributed. Printing the output of the above  result:

.. code::

    FeatureAttributionOutput({
        sequence_attributions: list with 1 elements of type GradientFeatureAttributionSequenceOutput: [
            GradientFeatureAttributionSequenceOutput({
                source: list with 13 elements of type TokenWithId:[
                    '▁Hello', '▁world', ',', '▁here', '\'', 's', '▁the', '▁In', 'se', 'q', '▁library', '!', '</s>'
                ],
                target: list with 12 elements of type TokenWithId:[
                    '▁Bonjour', '▁le', '▁monde', ',', '▁voici', '▁la', '▁bibliothèque', '▁Ins', 'e', 'q', '!', '</s>'
                ],
                source_attributions: torch.float32 tensor of shape [13, 12, 512] on cpu,
                target_attributions: None,
                step_scores: {},
                sequence_scores: None,
                attr_pos_start: 0,
                attr_pos_end: 12,
            })
        ],
        step_attributions: None,
        info: {
            ...
        }
    })

The tensor in the ``source_attribution`` field contains one attribution score per model's hidden size (512 here) for every source token (13 in this example, shown in ``source``) at every step of generation (12, shown in ``target``). The :class:`~inseq.data.GradientFeatureAttributionSequenceOutput` is a special class derived by the regular :class:`~inseq.data.FeatureAttributionSequenceOutput` that would automatically handle the last dimension of attribution tensors by summing an L2-normalizing via an :class:`~inseq.data.Aggregator`. This allows using the ``out.show`` function and automatically obtaining a 2-dimensional attribution map despite the original attribution tensor is 3-dimensional.
