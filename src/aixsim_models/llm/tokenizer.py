from typing import (
    Optional,
    AbstractSet,
    Union,
    Literal,
    Collection,
    Sequence,
    cast,
    Iterator,
)
import logging

from transformers import AutoTokenizer

from torchtitan.datasets.tokenizer import Tokenizer, TikTokenizer


logger = logging.getLogger(__name__)


class HFTokenizer:
    special_tokens: dict[str, int]
    num_reserved_special_tokens = ...
    pat_str = ...

    def __init__(self, tokenizer_path: str):
        self.model = AutoTokenizer.from_pretrained(tokenizer_path)
        self._n_words = self.model.vocab_size

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Optional[Union[Literal["all"], AbstractSet[str]]] = None,
        disallowed_special: Optional[Union[Literal["all"], Collection[str]]] = None,
    ) -> list[int]:
        assert type(s) is str
        if allowed_special is not None:
            raise NotImplementedError
        if disallowed_special is not None:
            raise NotImplementedError

        t = self.model.encode(s, add_special_tokens=False)
        if bos:
            t.insert(0, self.model.bos_token_id)
        if eos:
            t.append(self.model.eos_token_id)

        return t

    def decode(self, t: Sequence[int]) -> str:
        return self.model.decode(cast(list[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]

    @property
    def n_words(self) -> int:
        return self._n_words


def build_tokenizer(tokenizer_type: str, tokenizer_path: str) -> Tokenizer:
    if tokenizer_type == "tiktoken":
        tok = TikTokenizer(tokenizer_path)
    elif tokenizer_type == "hf":
        tok = HFTokenizer(tokenizer_path)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    logger.info(f"Building {tokenizer_type} tokenizer from {tokenizer_path}")
    return tok
