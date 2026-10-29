# Data and model provenance

TAT-QA is a dataset of financial tables, paragraphs, questions, and annotations. The
[dataset project](https://nextplusplus.github.io/TAT-QA/) explicitly releases its dataset
under **CC BY 4.0**. Its code repository has a separate MIT license. Cite:

Fengbin Zhu et al. (2021). *TAT-QA: A Question Answering Benchmark on a Hybrid of
Tabular and Textual Content in Finance*. ACL. https://aclanthology.org/2021.acl-long.254/

Downloads are pinned to repository commit
`870accc41953dcde885aabeb963d94aabdc0fbc3`. Each download creates a receipt containing
the URL, retrieval time, byte count, and SHA-256. Reusing a cached file verifies its
hash. The development snapshot has 278 contexts and 1,668 questions, of which only
numeric arithmetic questions are eligible for this project's current evaluator.
The raw dataset stays outside version control. Reports include IDs and evaluation
annotations; preserve the original dataset attribution if redistributing them.

The optional comparison models are [Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)
and [Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF),
both Q4_K_M quantizations with Apache 2.0 model licenses. The downloaded weights are
491,400,032 bytes and 1,117,320,736 bytes, respectively. Exact revisions and SHA-256
hashes are preserved in `reports/runtime-provenance.json`.
[llama.cpp](https://github.com/ggml-org/llama.cpp) is MIT-licensed;
the locally verified macOS arm64 runtime is release `b11669` (semantic release v0.7.0).
Weights and binaries are ignored and must not be included in the software license.

Public benchmark data may have appeared in a language model's pretraining corpus.
These experiments evaluate the implemented system; they do not establish uncontaminated
generalization or commercial suitability of underlying financial report content.

The v3 reasoning model is the official [Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF)
Q4_K_M quantization, under Apache 2.0, pinned to revision
`bc640142c66e1fdd12af0bd68f40445458f3869b`. Its size is 2,497,280,256 bytes and
SHA-256 is `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`.
The three optional model files occupy about 4.1 GB combined; only the 4B file is
needed for v3. The experiment uses nonthinking Qwen3 generation with all relevant
sampling settings recorded in the frozen policy artifact and runtime fingerprints.
The local verified launcher checks model bytes before starting inference.
