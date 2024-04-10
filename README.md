# Red-Teaming Language Models with DSPy

We use the the power of [DSPy](https://github.com/stanfordnlp/dspy), a framework for structuring and optimizing language model programs, to red-team language models. 

To our knowledge, this is the first attempt at using any auto-prompting *framework* to perform the red-teaming task. We accomplish this using a *deep* language program with several layers of alternating `Attack` and `Refine` modules in the following optimization loop:

<figure style="text-align: center;">
  <img src="/images/DSPy-Redteam.png" alt="Overview of DSPy for red-teaming" style="margin: 0 auto; margin-bottom: 20px; display: block;">
  <figcaption><i>Figure 1: Overview of DSPy for red-teaming. The DSPy MIPRO optimizer, guided by a LLM as a judge, compiles our language program into an effective red-teamer against Vicuna.</i></figcaption>
</figure>


<br>
This is also probably the deepest architecture in public optimized with DSPy to date.

The following Table demonstrates the effectiveness of the chosen architecture, as well as the benefit of DSPy compilation:

<div align="center">

| **Architecture** | **ASR** |
|:------------:|:----------:|
| None (Raw Input)       |   10%   |
| Architecture (5 Layer)   | 26%   |
| Architecture (5 Layer) + Optimization  | 44%   |

<em>Table 1: ASR with raw harmful inputs, un-optimized architecture, and architecture post DSPy compilation.</em>
</div>

With *no specific prompt engineering*, we are able to achieve an Attack Success Rate of 44%, 4x over the baseline. This is by no means the SOTA, but considering how we essentially spent no effort designing the architecture and prompts, and considering how we just used an off-the-shelf optimizer with almost no hyperparameter tuning (except to fit compute constraints), we think it is pretty exciting that we can achieve this result!

Full exposition on the [Haize Labs blog](https://blog.haizelabs.com/posts/dspy/).