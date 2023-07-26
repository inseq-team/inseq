..
    Copyright 2023 The Inseq Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Step Functions
=======================================================================================================================

.. currentmodule:: inseq.attr.step_functions

Step Functions Default arguments
-----------------------------------------------------------------------------------------------------------------------

The default arguments passed to all step functions are collected in the :class:`StepFunctionArgs` class.

.. autoclass:: StepFunctionArgs

Pre-registered Step Functions
-----------------------------------------------------------------------------------------------------------------------

The following functions can be used out-of-the-box as attribution targets or step functions in the :meth:`inseq.models.AttributionModel.attribute` function call simply by passing their string identifier (function name minus the ``_fn`` suffix).

.. autofunction:: logit_fn

.. autofunction:: probability_fn

.. autofunction:: entropy_fn

.. autofunction:: crossentropy_fn

.. autofunction:: perplexity_fn

.. autofunction:: contrast_prob_fn

.. autofunction:: pcxmi_fn

.. autofunction:: kl_divergence_fn

.. autofunction:: contrast_prob_diff_fn

.. autofunction:: mc_dropout_prob_avg_fn

.. autofunction:: top_p_size_fn
