..
    Copyright 2021 The Inseq Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Feature Attribution
=======================================================================================================================


.. autoclass:: inseq.attr.FeatureAttribution
    :members:

Gradient Attribution Methods
-----------------------------------------------------------------------------------------------------------------------

.. autoclass:: inseq.attr.feat.GradientAttributionRegistry
    :members:


.. autoclass:: inseq.attr.feat.DeepLiftAttribution
    :members:


.. warning::
    The DiscretizedIntegratedGradientsAttribution class is currently exhibiting inconsistent behavior, so usage should be limited until further notice. See PR `# 114 <https://github.com/inseq-team/inseq/pull/114>`__ for additional info.

.. autoclass:: inseq.attr.feat.DiscretizedIntegratedGradientsAttribution
    :members:


.. autoclass:: inseq.attr.feat.IntegratedGradientsAttribution
    :members:


.. autoclass:: inseq.attr.feat.InputXGradientAttribution
    :members:


.. autoclass:: inseq.attr.feat.SaliencyAttribution
    :members:


Layer Attribution Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. autoclass:: inseq.attr.feat.LayerIntegratedGradientsAttribution
    :members:


.. autoclass:: inseq.attr.feat.LayerGradientXActivationAttribution
    :members:


.. autoclass:: inseq.attr.feat.LayerDeepLiftAttribution
    :members:


Attention Attribution Methods
-----------------------------------------------------------------------------------------------------------------------

.. autoclass:: inseq.attr.feat.AttentionAttributionRegistry
    :members:


.. autoclass:: inseq.attr.feat.AttentionAttribution
    :members:
