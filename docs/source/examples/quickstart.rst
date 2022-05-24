.. Quickstart to public methods and common use-cases of the Inseq library

    Copyright 2021 The Inseq Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

###################################
Getting started with Inseq |:bug:|
###################################

The Inseq library is a Pytorch-based toolkit inteded to wrap some common use-cases in the study of sequence-to-sequence NN-based NLP models for model interpretability purposes. At the moment, the library supports the following set of models and techniques:

**Models**

- All the models made available through the `AutoModelForSeq2SeqLM <https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSeq2SeqLM>`_ interface of the |:hugging_face:| `transformers <https://github.com/huggingface/transformers>`_ library (among others, `T5 <https://huggingface.co/docs/transformers/model_doc/t5>`_, `Bart <https://huggingface.co/docs/transformers/model_doc/bart>`_ and all >1000 `MarianNMT <https://huggingface.co/docs/transformers/model_doc/marian>`_ variants) can be used in combination with the feature attribution methods.

**Interpretability Methods**

- Feature attribution for sequence generation can be performed by using the :class:`~inseq.AttributionModel.attribute` interface. At the moment, only gradient-based feature attribution methods sourced from the `Captum <https://captum.ai>`_ library are available, but other popular occlusion and attention-based techniques will soon follow. The list of all available methods can be obtained by using the `inseq.list_feature_attribution_methods` method. Each method either points to its original implementation, or is thoroughly documented in its docstring.

The ``AttributionModel`` class
===================================

The `inseq.AttributionModel` class is `torch.nn.Module` intended to seamlessly wrap any sequence-to-sequence Pytorch model to enable its interpretability. More specifically, the class adds the following capabilities to the wrapped model:

- An `AttributionModel.load` method to load the weights of the wrapped model from a saved checkpoint, locally or remotely.

- An `AttributionModel.attribute` method used to perform feature attribution using the loaded model.

- Multiple

The `attribute` method
===================================

The `attribute` method provides a centralized access to **sequential feature attribution** o
