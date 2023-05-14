..
    Copyright 2022 The Inseq Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

#####
FAQ
#####

▶️ **What is the difference between Inseq and other tools like** `transformers-interpret <https://github.com/cdpierse/transformers-interpret>`__ **and** `ecco <https://ecco.readthedocs.io/en/main/>`__ **?**

The main difference is that Inseq main focus is on sequence generation models. ``transformers-interpret`` currently does not support attribution for causal or sequence-to-sequence generation tasks, so it cannot be used for tasks like translation, summarization and text generation which are native to Inseq. ``ecco`` is a more general library that supports attribution for a wide range of tasks, and also includes some support for causal and sequence-to-sequence models. However, Inseq is designed to provide an improved experience for the generation case by providing advanced features such as batched generation attribution, custom function attribution and attribution aggregation.

▶️ **How can I evaluate the attributions produced by Inseq? Do you plan to include interpretability metrics like faithfulness and plausibility in your library?**

At the moment we don't plan to include interpretability metrics in Inseq to avoid dispersing the focus of the library. This said, are currently planning a collaboration with the development team of the `ferret <https://ferret.readthedocs.io/en/latest/index.html>`__ library to enable the evaluation of Inseq's attributions with the metrics implemented in ferret.

▶️ **I have a sequence generation model that is not supported by 🤗 transformers. Can I still use Inseq with it?**

At the moment 🤗 transformers is the only high-level framework supported by Inseq. However, Inseq is designed to be framework-agnostic and other frameworks can be supported in the future. If you are interested in contributing to Inseq, please raise an issue on GitHub.

▶️ **Why the name Inseq? What does the logo represent?**

Inseq (ɪn'sek) is short for "**in**\terpretability for **seq**\uence generation models". The red rampant caterpillar was selected as logo for the library playing on the near-homophone word "insect", and because it is visually similar to source-side feature attribution maps with high saliency scores for aligned words on the main diagonal.
