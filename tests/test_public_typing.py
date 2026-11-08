"""Every type in a public signature must be importable from a public module.

Consumers annotate the values they pass to (and get back from) yaozarrs.  Each
name used in a public signature must therefore be one of: a builtin, a public
name from the stdlib or another package, a public yaozarrs class/model, or a
name exported by `yaozarrs.typing` (type aliases, which exist only for type
checkers).  An alias defined in a private module and then used in a public
signature (e.g. `CompressionName` in `write_image`) is exactly what this guards
against.

Annotations are strings (`from __future__ import annotations`), and the aliases
don't exist at runtime, so this is a static (ast-based) check of the source,
seen the way a type checker sees it: `if TYPE_CHECKING:` is taken to be true.
"""

from __future__ import annotations

import ast
import builtins
import textwrap
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import yaozarrs.typing

if TYPE_CHECKING:
    from collections.abc import Iterator

SRC = Path(__file__).parent.parent / "src"
TYPEVAR_FACTORIES = {"TypeVar", "ParamSpec", "TypeVarTuple"}


def _is_private(dotted: str) -> bool:
    return any(p.startswith("_") and not p.startswith("__") for p in dotted.split("."))


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _type_checking_view(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Statements of `body` as a type checker sees them (TYPE_CHECKING is True)."""
    for stmt in body:
        if isinstance(stmt, ast.If):
            if _is_type_checking(stmt.test):
                yield from _type_checking_view(stmt.body)
            else:  # (e.g. a sys.version_info check: either branch may apply)
                yield from _type_checking_view(stmt.body)
                yield from _type_checking_view(stmt.orelse)
        elif isinstance(stmt, ast.Try):
            yield from _type_checking_view(stmt.body)
            for handler in stmt.handlers:
                yield from _type_checking_view(handler.body)
            yield from _type_checking_view(stmt.orelse)
            yield from _type_checking_view(stmt.finalbody)
        else:
            yield stmt


@dataclass(frozen=True)
class Ref:
    """What a name in an annotation refers to."""

    target: str  # fully qualified, e.g. "yaozarrs.v05._image.Image" or "zarr.Array"
    kind: str  # builtin | external | module | class | function | alias | typevar
    node: ast.AST | None = field(default=None, compare=False, repr=False)


@dataclass
class Module:
    name: str
    path: Path

    @cached_property
    def tree(self) -> ast.Module:
        return ast.parse(self.path.read_text(encoding="utf-8"), str(self.path))

    @property
    def package(self) -> str:
        if self.path.name == "__init__.py":
            return self.name
        return self.name.rpartition(".")[0]

    @cached_property
    def bindings(self) -> dict[str, list[ast.stmt]]:
        """Top-level statements binding each name (several for overloads)."""
        out: dict[str, list[ast.stmt]] = {}
        for stmt in _type_checking_view(self.tree.body):
            for name in _bound_names(stmt):
                out.setdefault(name, []).append(stmt)
        return out

    @cached_property
    def all(self) -> list[str] | None:
        """`__all__`, as a type checker sees it (None if not defined)."""
        names: list[str] | None = None
        for stmt in _type_checking_view(self.tree.body):
            if isinstance(stmt, ast.Assign | ast.AnnAssign | ast.AugAssign):
                targets = (
                    stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                )
                is_all = any(getattr(t, "id", None) == "__all__" for t in targets)
                if is_all and stmt.value is not None:
                    value = ast.literal_eval(stmt.value)
                    if isinstance(stmt, ast.AugAssign):
                        names = [*(names or []), *value]
                    else:
                        names = list(value)
        return names

    def resolve_relative(self, node: ast.ImportFrom) -> str:
        if not node.level:
            return node.module or ""
        base = self.package.rsplit(".", node.level - 1)[0]
        return f"{base}.{node.module}" if node.module else base


def _bound_names(stmt: ast.stmt) -> Iterator[str]:
    if isinstance(stmt, ast.Import):
        for alias in stmt.names:
            yield alias.asname or alias.name.partition(".")[0]
    elif isinstance(stmt, ast.ImportFrom):
        for alias in stmt.names:
            yield alias.asname or alias.name
    elif isinstance(stmt, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
        yield stmt.name
    elif isinstance(stmt, ast.TypeAlias):
        yield stmt.name.id
    elif isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                yield target.id
    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        yield stmt.target.id


class Package:
    """Static view of a package's source tree."""

    def __init__(self, src: Path, name: str) -> None:
        self.modules: dict[str, Module] = {}
        for path in sorted((src / name).rglob("*.py")):
            parts = list(path.relative_to(src).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            mod = ".".join(parts)
            self.modules[mod] = Module(mod, path)
        self.name = name

    def owns(self, dotted: str) -> bool:
        return dotted == self.name or dotted.startswith(f"{self.name}.")

    @cached_property
    def public_modules(self) -> list[Module]:
        return [m for n, m in self.modules.items() if not _is_private(n)]

    # ------------------------- name resolution -------------------------

    def resolve(
        self, module: str, name: str, _seen: frozenset[str] = frozenset()
    ) -> Ref:
        """Resolve `name`, as accessed on `module`, to where it is defined."""
        dotted = f"{module}.{name}"
        if not self.owns(module):
            return Ref(dotted, "external")
        if dotted in self.modules:
            return Ref(dotted, "module")
        if dotted in _seen:
            raise LookupError(f"import cycle resolving {dotted}")
        stmts = self.modules[module].bindings.get(name)
        if not stmts:
            raise LookupError(f"{name!r} is not defined in {module}")
        stmt = stmts[-1]
        mod = self.modules[module]
        if isinstance(stmt, ast.ImportFrom):
            source = mod.resolve_relative(stmt)
            alias = next(a for a in stmt.names if (a.asname or a.name) == name)
            return self.resolve(source, alias.name, _seen | {dotted})
        if isinstance(stmt, ast.Import):
            alias = next(
                a for a in stmt.names if (a.asname or a.name.partition(".")[0]) == name
            )
            target = alias.name if alias.asname else name
            kind = "module" if self.owns(target) else "external"
            return Ref(target, kind)
        if isinstance(stmt, ast.ClassDef):
            return Ref(dotted, "class", stmt)
        if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
            return Ref(dotted, "function", stmt)
        value = getattr(stmt, "value", None)
        if (
            isinstance(value, ast.Call)
            and getattr(value.func, "id", getattr(value.func, "attr", None))
            in TYPEVAR_FACTORIES
        ):
            return Ref(dotted, "typevar", stmt)
        return Ref(dotted, "alias", stmt)

    def resolve_expr(self, module: str, expr: ast.expr) -> Ref:
        """Resolve a Name or (dotted) Attribute used in `module`."""
        if isinstance(expr, ast.Name):
            if expr.id in self.modules[module].bindings:
                return self.resolve(module, expr.id)
            if hasattr(builtins, expr.id):
                return Ref(expr.id, "builtin")
            raise LookupError(f"{expr.id!r} is not defined in {module}")
        if isinstance(expr, ast.Attribute):
            base = self.resolve_expr(module, expr.value)
            if base.kind == "module":
                return self.resolve(base.target, expr.attr)
            if base.kind == "external":
                return Ref(f"{base.target}.{expr.attr}", "external")
            return Ref(f"{base.target}.{expr.attr}", "attribute")
        raise TypeError(f"cannot resolve {ast.unparse(expr)}")

    def annotation_refs(self, module: str, expr: ast.expr | None) -> Iterator[Ref]:
        """Every name referenced by the annotation `expr` (written in `module`)."""
        if expr is None:
            return
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, str):  # a forward reference
                parsed = ast.parse(expr.value.strip(), mode="eval").body
                yield from self.annotation_refs(module, parsed)
        elif isinstance(expr, ast.Name | ast.Attribute):
            yield self.resolve_expr(module, expr)
        elif isinstance(expr, ast.Subscript):
            origin = self.resolve_expr(module, expr.value)
            yield origin
            origin_name = origin.target.rpartition(".")[2]
            if origin.kind == "external" and origin_name == "Literal":
                return  # (strings in Literal[...] are values, not forward refs)
            if origin.kind == "external" and origin_name == "Annotated":
                assert isinstance(expr.slice, ast.Tuple)
                yield from self.annotation_refs(module, expr.slice.elts[0])
                return
            yield from self.annotation_refs(module, expr.slice)
        elif isinstance(expr, ast.BinOp):  # X | Y
            yield from self.annotation_refs(module, expr.left)
            yield from self.annotation_refs(module, expr.right)
        elif isinstance(expr, ast.Tuple | ast.List):  # (e.g. Callable[[X], Y])
            for elt in expr.elts:
                yield from self.annotation_refs(module, elt)
        else:
            raise TypeError(f"unexpected annotation: {ast.unparse(expr)}")

    # -------------------------- the public API -------------------------

    @cached_property
    def public_targets(self) -> set[str]:
        """Definitions that can be imported from a public module."""
        out = set()
        for mod in self.public_modules:
            for name in self._exports(mod):
                out.add(self.resolve(mod.name, name).target)
        return out

    def _exports(self, mod: Module) -> list[str]:
        if mod.all is not None:
            return mod.all
        return [n for n in mod.bindings if not n.startswith("_")]

    def is_public(self, ref: Ref) -> bool:
        if ref.kind in ("builtin", "typevar"):
            return True
        if ref.kind in ("external", "module"):
            return not _is_private(ref.target)
        return ref.target in self.public_targets

    def api_items(self) -> Iterator[tuple[str, str, ast.AST]]:
        """(public name, module, node) for each public function/method/alias."""
        seen: set[str] = set()
        for mod in self.public_modules:
            for name in self._exports(mod):
                ref = self.resolve(mod.name, name)
                where = f"{mod.name}.{name}"
                if ref.target in seen or ref.kind in ("module", "external"):
                    continue
                seen.add(ref.target)
                def_mod, _, def_name = ref.target.rpartition(".")
                if ref.kind == "function":
                    for stmt in self.modules[def_mod].bindings[def_name]:
                        yield where, def_mod, stmt
                elif ref.kind == "class":
                    yield from self._class_methods(where, ref)
                elif ref.kind == "alias" and ref.node and self.owns(def_mod):
                    yield where, def_mod, ref.node

    def _class_methods(
        self, where: str, cls: Ref
    ) -> Iterator[tuple[str, str, ast.AST]]:
        """Public methods of `cls` (and inherited ones), where they're defined."""
        done: set[str] = set()
        for base in self._mro(cls):
            module = base.target.rpartition(".")[0]
            assert isinstance(base.node, ast.ClassDef)
            methods: dict[str, list[ast.stmt]] = {}
            for stmt in _type_checking_view(base.node.body):
                if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
                    methods.setdefault(stmt.name, []).append(stmt)
            for name, stmts in methods.items():
                is_dunder = name.startswith("__") and name.endswith("__")
                if name in done or (name.startswith("_") and not is_dunder):
                    continue
                done.add(name)
                for stmt in stmts:
                    yield f"{where}.{name}", module, stmt

    def _mro(self, cls: Ref) -> Iterator[Ref]:
        """`cls` and its yaozarrs base classes (depth-first, left to right)."""
        yield cls
        assert isinstance(cls.node, ast.ClassDef)
        module = cls.target.rpartition(".")[0]
        for base in cls.node.bases:
            expr = base.value if isinstance(base, ast.Subscript) else base
            ref = self.resolve_expr(module, expr)
            if ref.kind == "class":
                yield from self._mro(ref)

    def problems(self) -> list[str]:
        out = []
        for where, module, node in self.api_items():
            for expr, default in _annotations(node):
                try:
                    refs = list(self.annotation_refs(module, expr))
                    # (the type of a sentinel default, e.g. `x: int | _Unset = _UNSET`,
                    # needn't be public: callers get that default by omitting `x`)
                    sentinels = {
                        self.resolve_expr(module, n).target
                        for n in ast.walk(default or ast.Constant(None))
                        if isinstance(n, ast.Name)
                    }
                except LookupError as e:
                    out.append(f"{where}: cannot resolve `{ast.unparse(expr)}` ({e})")
                    continue
                for ref in refs:
                    if not self.is_public(ref) and ref.target not in sentinels:
                        out.append(
                            f"{where}: `{ast.unparse(expr)}` uses {ref.target}, "
                            "which is not importable from a public module"
                        )
        return sorted(set(out))


def _is_type_alias(annotation: ast.expr) -> bool:
    return getattr(annotation, "id", getattr(annotation, "attr", None)) == "TypeAlias"


def _annotations(node: ast.AST) -> Iterator[tuple[ast.expr, ast.expr | None]]:
    """(annotation, default value) pairs of a function, or what an alias aliases."""
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        args = node.args
        positional = [*args.posonlyargs, *args.args]
        defaults = [None] * (len(positional) - len(args.defaults)) + args.defaults
        params = [
            *zip(positional, defaults, strict=True),
            *zip(args.kwonlyargs, args.kw_defaults, strict=True),
            (args.vararg, None),
            (args.kwarg, None),
        ]
        for arg, default in params:
            if arg is not None and arg.annotation is not None:
                yield arg.annotation, default
        if node.returns is not None:
            yield node.returns, None
    elif isinstance(node, ast.AnnAssign) and not _is_type_alias(node.annotation):
        yield node.annotation, None  # (an exported variable)
    elif isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None:
        # an exported alias: what it aliases must be public too
        yield node.value, None
    elif isinstance(node, ast.TypeAlias):
        yield node.value, None


def test_public_signatures_use_public_types() -> None:
    problems = Package(SRC, "yaozarrs").problems()
    assert not problems, "\n".join(problems)


# --------------------------- the checker itself ---------------------------

# A private alias, used in public signatures: what broke ome-writers in v0.4.0
# (`CompressionName` moved between private modules, and its import with it).
BAD_PACKAGE = {
    "__init__.py": "",
    "_arrays.py": """
        from typing import TYPE_CHECKING, Literal

        if TYPE_CHECKING:
            CompressionName = Literal["zstd", "none"]
        """,
    "_unset.py": """
        import enum

        class _Unset(enum.Enum):
            UNSET = "unset"
        """,
    "_base.py": """
        from __future__ import annotations

        from typing import TYPE_CHECKING

        from pkg._unset import _Unset

        if TYPE_CHECKING:
            from pkg._arrays import CompressionName

        class _Hidden: ...

        class BuilderBase:
            def write(self, compression: CompressionName) -> _Hidden: ...
            def add(self, compression: CompressionName | _Unset = _Unset.UNSET): ...
            def _private(self, compression: CompressionName) -> None: ...
        """,
    "write/__init__.py": """
        from ._write import Builder, write_image

        __all__ = ["Builder", "write_image"]
        """,
    "write/_write.py": """
        from __future__ import annotations

        from typing import TYPE_CHECKING, Literal

        from pkg._base import BuilderBase

        if TYPE_CHECKING:
            from pkg._arrays import CompressionName

        class Builder(BuilderBase): ...

        def write_image(
            compression: CompressionName = "zstd", mode: Literal["Mode"] = "Mode"
        ) -> None: ...
        """,
}
# The fix: define the alias in a public module, for type checkers only
GOOD_TYPING_MODULE = """
    from __future__ import annotations

    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from typing import Literal

        from pkg._base import _Hidden as Hidden

        __all__ = ["CompressionName", "Hidden"]

        CompressionName = Literal["zstd", "none"]
    else:
        __all__ = []
    """
GOOD_PACKAGE = {
    **BAD_PACKAGE,
    "typing.py": GOOD_TYPING_MODULE,
    "_arrays.py": "",
    "_base.py": BAD_PACKAGE["_base.py"].replace("pkg._arrays", "pkg.typing"),
    "write/_write.py": BAD_PACKAGE["write/_write.py"].replace(
        "pkg._arrays", "pkg.typing"
    ),
}


def _package(root: Path, files: dict[str, str]) -> Package:
    for name, source in files.items():
        path = root / "pkg" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return Package(root, "pkg")


def test_checker_catches_private_aliases(tmp_path: Path) -> None:
    uses = "which is not importable from a public module"
    assert _package(tmp_path, BAD_PACKAGE).problems() == [
        f"pkg.write.Builder.add: `CompressionName | _Unset` uses "
        f"pkg._arrays.CompressionName, {uses}",
        f"pkg.write.Builder.write: `CompressionName` uses "
        f"pkg._arrays.CompressionName, {uses}",
        f"pkg.write.Builder.write: `_Hidden` uses pkg._base._Hidden, {uses}",
        f"pkg.write.write_image: `CompressionName` uses "
        f"pkg._arrays.CompressionName, {uses}",
    ]


def test_checker_accepts_public_aliases(tmp_path: Path) -> None:
    assert _package(tmp_path, GOOD_PACKAGE).problems() == []


# -------------------------- yaozarrs.typing at runtime --------------------------


def test_typing_names_are_type_checking_only() -> None:
    names = Package(SRC, "yaozarrs").modules["yaozarrs.typing"].all
    assert names and "CompressionName" in names
    assert set(names) == yaozarrs.typing._TYPE_CHECKING_ONLY
    for name in names:
        with pytest.raises(ImportError, match="if TYPE_CHECKING:"):
            getattr(yaozarrs.typing, name)

    with pytest.raises(ImportError, match="'CompressionName' cannot be imported"):
        from yaozarrs.typing import CompressionName  # noqa: F401

    # a star import gets nothing, and other names behave as usual (e.g. for tools
    # that probe a module for hooks, like pytest's `pytest_plugins`)
    namespace: dict[str, object] = {}
    exec("from yaozarrs.typing import *", namespace)
    assert set(namespace) == {"__builtins__"}
    assert getattr(yaozarrs.typing, "pytest_plugins", None) is None
    with pytest.raises(AttributeError):
        yaozarrs.typing.CompresionName  # noqa: B018


def test_typing_docstring_example_runs() -> None:
    """The documented usage works at runtime (the import never runs)."""
    assert yaozarrs.typing.__doc__
    code = yaozarrs.typing.__doc__.split("```python\n")[1].split("```")[0]
    assert "if TYPE_CHECKING:" in code
    exec(compile(code, "<yaozarrs.typing docstring>", "exec"), {})
