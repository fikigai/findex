import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from findex.index import Index
from findex.tokenizer import tokenize


class QuerySyntaxError(ValueError):
    """Помилка синтаксису пошукового запиту."""


class QueryNode(ABC):
    @abstractmethod
    def evaluate(self, index: Index) -> set[str]:
        ...

    @abstractmethod
    def extract_terms(self) -> list[str]:
        ...

    def __and__(self, other: "QueryNode") -> "AndNode":
        return AndNode(self, other)

    def __or__(self, other: "QueryNode") -> "OrNode":
        return OrNode(self, other)

    def __invert__(self) -> "NotNode":
        return NotNode(self)


@dataclass(frozen=True)
class TermNode(QueryNode):
    term: str

    def evaluate(self, index: Index) -> set[str]:
        return {p.doc_id for p in index.index.get(self.term, [])}

    def extract_terms(self) -> list[str]:
        return [self.term]


@dataclass(frozen=True)
class PhraseNode(QueryNode):
    terms: tuple[str, ...]

    def __init__(self, terms: Sequence[str]) -> None:
        object.__setattr__(self, "terms", tuple(terms))

    def evaluate(self, index: Index) -> set[str]:
        if not self.terms:
            return set()
        postings_lists = [index.index.get(t, []) for t in self.terms]
        if any(not p for p in postings_lists):
            return set()

        doc_maps = [{p.doc_id: p.positions for p in plist} for plist in postings_lists]
        common_docs = set(doc_maps[0].keys())
        for dmap in doc_maps[1:]:
            common_docs.intersection_update(dmap.keys())

        matching_docs: set[str] = set()
        for doc_id in common_docs:
            positions_list = [dmap[doc_id] for dmap in doc_maps]
            for p0 in positions_list[0]:
                if all((p0 + k) in positions_list[k] for k in range(1, len(positions_list))):
                    matching_docs.add(doc_id)
                    break
        return matching_docs

    def extract_terms(self) -> list[str]:
        return list(self.terms)


@dataclass(frozen=True)
class AndNode(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[str]:
        return self.left.evaluate(index) & self.right.evaluate(index)

    def extract_terms(self) -> list[str]:
        return self.left.extract_terms() + self.right.extract_terms()


@dataclass(frozen=True)
class OrNode(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[str]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def extract_terms(self) -> list[str]:
        return self.left.extract_terms() + self.right.extract_terms()


@dataclass(frozen=True)
class NotNode(QueryNode):
    node: QueryNode

    def evaluate(self, index: Index) -> set[str]:
        all_docs = set(index.docs.keys())
        return all_docs - self.node.evaluate(index)

    def extract_terms(self) -> list[str]:
        return []


_TOKEN_RE = re.compile(r'\(|\)|"[^"]+"|[^\s()"]+')


class QueryParser:
    """Парсер рекурсивного спуску для булевих та фразових запитів."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self) -> str:
        tok = self.peek()
        if tok is None:
            raise QuerySyntaxError("Неочікуваний кінець запиту")
        self.pos += 1
        return tok

    def parse(self) -> QueryNode | None:
        if not self.tokens:
            return None
        node = self.parse_or()
        if self.peek() is not None:
            raise QuerySyntaxError(f"Неочікуваний токен у запиті: {self.peek()}")
        return node

    def parse_or(self) -> QueryNode:
        node = self.parse_and()
        while self.peek() == "OR":
            self.consume()
            right = self.parse_and()
            node = OrNode(node, right)
        return node

    def parse_and(self) -> QueryNode:
        node = self.parse_not()
        while self.peek() is not None and self.peek() not in ("OR", ")"):
            if self.peek() == "AND":
                self.consume()
            right = self.parse_not()
            node = AndNode(node, right)
        return node

    def parse_not(self) -> QueryNode:
        if self.peek() == "NOT":
            self.consume()
            return NotNode(self.parse_primary())
        return self.parse_primary()

    def parse_primary(self) -> QueryNode:
        tok = self.peek()
        if tok is None:
            raise QuerySyntaxError("Очікувався терм або вираз у запиті")
        if tok in ("AND", "OR", ")"):
            raise QuerySyntaxError(f"Некоректне розташування оператора: {tok}")
        if tok == "(":
            self.consume()
            if self.peek() == ")":
                raise QuerySyntaxError("Порожні дужки у запиті")
            node = self.parse_or()
            if self.peek() != ")":
                raise QuerySyntaxError("Незакрита дужка '(' у запиті")
            self.consume()
            return node
        if tok.startswith('"'):
            if not tok.endswith('"') or len(tok) < 2:
                raise QuerySyntaxError("Незакриті лапки у фразовому запиті")
            self.consume()
            phrase_terms = list(tokenize(tok[1:-1]))
            return PhraseNode(phrase_terms)

        raw = self.consume()
        if '"' in raw:
            raise QuerySyntaxError("Незакриті лапки у запиті")
        terms = list(tokenize(raw))
        return TermNode(terms[0] if terms else raw.casefold())


def parse_query(query_str: str) -> QueryNode | None:
    if query_str.count('"') % 2 != 0:
        raise QuerySyntaxError("Непарна кількість лапок у запиті")
    raw_tokens = _TOKEN_RE.findall(query_str)
    return QueryParser(raw_tokens).parse()