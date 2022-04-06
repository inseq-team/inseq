..
    Copyright 2022 The Inseq Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

##########################################
Comparing Attributions with PairAggregator
##########################################

Inseq support minimal pair analysis via the `PairAggregator </main_classes/data_classes.html#pairaggregator>`_ component.

Here is an example of using ``PairAggregator`` to produce a heatmap to visualize the score difference between two ``FeatureAttributionSequenceOutput`` objects:

.. code-block:: python

    import inseq
    from inseq.data.aggregator import AggregatorPipeline, ContiguousSpanAggregator, SequenceAttributionAggregator, PairAggregator

    # Load the EN-FR translation model and attach the IG method
    model = inseq.load_model("Helsinki-NLP/opus-mt-en-fr", "integrated_gradients")

    # Perform the attribution with forced decoding. Return convergence deltas, probabilities and target attributions.
    out = model.attribute(
        [
            "The manager told the hairdresser that the haircut he made her was terrible.",
            "The manager told the hairdresser that the haircut he made her was terrible.",
        ],
        [
            "Le gérant a dit au coiffeur que la coupe de cheveux qu'il lui a faite était terrible.",
            "La gérante a dit au coiffeur que la coupe de cheveux qu'il lui a faite était terrible.",
        ],
        n_steps=300,
        return_convergence_delta=True,
        attribute_target=True,
        output_step_probabilities=True,
        internal_batch_size=100,
        include_eos_baseline=False,
    )

    # Aggregation pipeline composed by two steps:
    # 1. Aggregate contiguous tokens across all attribution dimensions
    # 2. Aggregate the last dimension of the neuron-level attribution to make it token-level
    squeezesum = AggregatorPipeline([ContiguousSpanAggregator, SequenceAttributionAggregator])

    # Simply aggregate over the last dimension for the masculine variant
    masculine = out.sequence_attributions[0].aggregate(aggregator=SequenceAttributionAggregator)

    # For the feminine variant, we also use the contiguous span aggregator to merge "▁gérant" "e"
    # in a single token to match masculine shape
    feminine = out.sequence_attributions[1].aggregate(aggregator=squeezesum, target_spans=(1, 3))

    # Take the diff of the scores of the two attribution and show it
    masculine.show(aggregator=PairAggregator, paired_attr=feminine)


.. raw:: html

    <div class="html-example">
        <iframe frameborder="0" scale="0.75" src="/_static/pair_comparison.htm"></iframe>
    </div>
