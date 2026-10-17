"""Compatibility shim for sentence-transformers 6.1 + transformers 5.17 on text-only Gemma 3.

ST's Transformer module calls AutoProcessor.from_pretrained(), which maps model_type
`gemma3_text` to the multimodal Gemma3Processor and fails because the Harrier repo has no
preprocessor_config.json. For a text-only model the processor is just the tokenizer, so
fall back to AutoTokenizer on that specific failure. This changes no tokenization or
model math (tokenizer.json / tokenizer_config.json are loaded exactly as before).

Import this module before constructing SentenceTransformer.
"""
import transformers
from transformers import AutoProcessor, AutoTokenizer

_orig = AutoProcessor.from_pretrained.__func__


def _from_pretrained(cls, name, *args, **kwargs):
    try:
        return _orig(cls, name, *args, **kwargs)
    except OSError as e:
        if "image processor" not in str(e) and "preprocessor_config" not in str(e):
            raise
        return AutoTokenizer.from_pretrained(name, *args, **kwargs)


AutoProcessor.from_pretrained = classmethod(_from_pretrained)
transformers.AutoProcessor.from_pretrained = AutoProcessor.from_pretrained
