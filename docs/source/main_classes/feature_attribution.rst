..
    Copyright 2021 The Inseq Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Attribution Methods
=======================================================================================================================


.. autoclass:: inseq.attr.FeatureAttribution
    :members:

Gradient-based Attribution Methods
-----------------------------------------------------------------------------------------------------------------------

.. autoclass:: inseq.attr.feat.GradientAttributionRegistry
    :members:


.. autoclass:: inseq.attr.feat.DeepLiftAttribution
    :members:


.. autoclass:: inseq.attr.feat.DiscretizedIntegratedGradientsAttribution
    :members:

.. autoclass:: inseq.attr.feat.GradientShapAttribution
    :members:


.. autoclass:: inseq.attr.feat.IntegratedGradientsAttribution
    :members:


.. autoclass:: inseq.attr.feat.InputXGradientAttribution
    :members:


.. autoclass:: inseq.attr.feat.SaliencyAttribution
    :members:


.. autoclass:: inseq.attr.feat.SequentialIntegratedGradientsAttribution
    :members:


Layer Attribution Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. autoclass:: inseq.attr.feat.LayerIntegratedGradientsAttribution
    :members:


.. autoclass:: inseq.attr.feat.LayerGradientXActivationAttribution
    :members:


.. autoclass:: inseq.attr.feat.LayerDeepLiftAttribution
    :members:


Internals-based Attribution Methods
-----------------------------------------------------------------------------------------------------------------------

.. autoclass:: inseq.attr.feat.InternalsAttributionRegistry
    :members:


.. autoclass:: inseq.attr.feat.AttentionWeightsAttribution
    :members:

Perturbation-based Attribution Methods
-----------------------------------------------------------------------------------------------------------------------

.. autoclass:: inseq.attr.feat.PerturbationAttributionRegistry
    :members:

.. autoclass:: inseq.attr.feat.OcclusionAttribution
    :members:

.. autoclass:: inseq.attr.feat.LimeAttribution
    :members:

.. autoclass:: inseq.attr.feat.ValueZeroingAttribution
    :members:

.. autoclass:: inseq.attr.feat.ReagentAttribution
    :members:

    .. automethod:: __init__

.. code:: python

    import inseq

    model = inseq.load_model(
        "gpt2-medium",
        "reagent",
        keep_top_n=5,
        stopping_condition_top_k=3,
        replacing_ratio=0.3,
        max_probe_steps=3000,
        num_probes=8
    )
    out = model.attribute("Super Mario Land is a game that developed by")
    out.show()
