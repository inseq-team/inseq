..
    Copyright 2024 The Inseq Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Inseq CLI
=======================================================================================================================

The Inseq CLI is a command line interface for the Inseq library. The CLI enables repeated attribution of individual 
examples and even entire 🤗 datasets directly from the console. See the available options by typing ``inseq -h`` in the 
terminal after installing the package.

Three commands are supported:

- ``inseq attribute``: Wrapper for enabling ``model.attribute`` usage in console.

- ``inseq attribute-dataset``: Extends ``attribute`` to full dataset using Hugging Face ``datasets.load_dataset`` API.

- ``inseq attribute-context``: Detects and attribute context dependence for generation tasks using the approach of `Sarti et al. (2023) <https://arxiv.org/abs/2310.0118>`__.

``attribute``
-----------------------------------------------------------------------------------------------------------------------

The ``attribute`` command enables attribution of individual examples directly from the console. The command takes the
following arguments:

.. autoclass:: inseq.commands.attribute.attribute_args.AttributeWithInputsArgs

``attribute-dataset``
-----------------------------------------------------------------------------------------------------------------------

The ``attribute-dataset`` command extends the ``attribute`` command to full datasets using the Hugging Face 
``datasets.load_dataset`` API. The command takes the following arguments:

.. autoclass:: inseq.commands.attribute_dataset.attribute_dataset_args.LoadDatasetArgs

.. autoclass:: inseq.commands.attribute.attribute_args.AttributeExtendedArgs

``attribute-context``
-----------------------------------------------------------------------------------------------------------------------

The ``attribute-context`` command detects and attributes context dependence for generation tasks using the approach of
`Sarti et al. (2023) <https://arxiv.org/abs/2310.0118>`__. The command takes the following arguments:

.. autoclass:: inseq.commands.attribute_context.attribute_context_args.AttributeContextArgs