from collections.abc import Iterator

class _Word:
    text: str
    upos: str
    feats: str | None
    head: int
    deprel: str
    id: int

class _Span:
    text: str
    type: str
    start_char: int
    end_char: int

class _Sentence:
    words: list[_Word]
    ents: list[_Span]
    def __iter__(self) -> Iterator[_Word]: ...

class _Document:
    sentences: list[_Sentence]
    ents: list[_Span]

class Pipeline:
    def __init__(self, lang: str = ..., processors: str = ..., download_method: object = ...) -> None: ...
    def __call__(self, text: str) -> _Document: ...

__version__: str
