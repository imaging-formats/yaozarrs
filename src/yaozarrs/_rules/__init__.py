"""Stable codes for the normative statements of the NGFF spec.

Each code (e.g. `YZ-AX-002`) identifies one normative statement in the NGFF spec, and
links to a docs page describing it; checks that go beyond the NGFF spec are `YZ-NNN`.
The registry of all codes lives in `rules.toml`. yaozarrs errors and warnings for spec
violations are being tagged with these codes, using `rule_error` and `warn_rule`.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

from pydantic_core import PydanticCustomError

from yaozarrs._errors import ValidationWarning
from yaozarrs._rules._model import Rule, SpecDoc, SpecSource
from yaozarrs._rules._registry import RULES

if TYPE_CHECKING:
    from typing import LiteralString

__all__ = [
    "RULES",
    "Rule",
    "RuleWarning",
    "SpecDoc",
    "SpecSource",
    "error_code",
    "get_rule",
    "rewrap_error",
    "rule_error",
    "rule_url",
    "warn_rule",
]


class RuleWarning(ValidationWarning):
    r"""A warning tied to a rule code.

    The message starts with `[<code>]`, so it can be filtered with e.g.
    `warnings.filterwarnings("ignore", message=r"\[YZ-PL-002\]")`.
    """

    def __init__(self, code: str, msg: str) -> None:
        self.code = code
        self.msg = msg
        self.url = rule_url(code)
        super().__init__(f"[{code}] {msg}\n  See {self.url}")

    def __reduce__(self) -> tuple[type[RuleWarning], tuple[str, str]]:
        # (for pickle and copy: `args` holds the formatted message, not these)
        return type(self), (self.code, self.msg)


def get_rule(code: str) -> Rule:
    return RULES[code]


def rule_url(code: str) -> str:
    return get_rule(code).url


def rule_error(code: str, msg: str) -> PydanticCustomError:
    """Return an error (to be raised in a pydantic validator) for rule `code`."""
    # no context is passed, so braces in `msg` are left alone by pydantic
    template = f"{msg}\n  See {rule_url(code)}"
    return PydanticCustomError(
        cast("LiteralString", code), cast("LiteralString", template)
    )


def error_code(e: Exception) -> str | None:
    """Return the rule code of error `e`, if it has one."""
    # (yaozarrs raises other custom errors too, e.g. `listItemsNotUnique`)
    if isinstance(e, PydanticCustomError) and e.type in RULES:
        return e.type
    return None


def rewrap_error(
    e: ValueError, prefix: str = "", suffix: str = "", code: str | None = None
) -> ValueError:
    """Return `e` with more context in its message, and optionally a different code."""
    msg = str(e)
    if old := error_code(e):
        msg = msg.removesuffix(f"\n  See {rule_url(old)}")
    msg = f"{prefix}{msg}{suffix}"
    if code := code or old:
        return rule_error(code, msg)
    if isinstance(e, PydanticCustomError):  # keep a non-rule error's type and context
        template = f"{prefix}{e.message_template}{suffix}"
        return PydanticCustomError(
            cast("LiteralString", e.type), cast("LiteralString", template), e.context
        )
    return ValueError(msg)


def warn_rule(code: str, msg: str, stacklevel: int = 2) -> None:
    """Emit a `RuleWarning` for rule `code`."""
    warnings.warn(RuleWarning(code, msg), stacklevel=stacklevel + 1)
