from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import List, Set
from findex.index import Index
from findex.tokenizer import tokenize


class QueryNode(ABC):
    @abstractmethod
    def evaluate(self, index: Index) -> Set[str]:
        pass

    @abstractmethod
    def extract_terms(self) -> List[str]:
        pass

    def __and__(self, other: "QueryNode") -> "AndNode":
        return AndNode(self, other)

    def __or__(self, other: "QueryNode") -> "OrNode":
        return OrNode(self, other)

    def __invert__(self) -> "NotNode":
        return NotNode(self)


@dataclass(frozen=True)
class TermNode(QueryNode):
    term: str

    def evaluate(self, index: Index) -> Set[str]:
        return {p.doc_id for p in index.index.get(self.term, [])}

    def extract_terms(self) -> List[str]:
        return [self.term]


@dataclass(frozen=True)
class PhraseNode(QueryNode):
    terms: tuple[str, ...]

    def __init__(self, terms: List[str] | tuple[str, ...]):
        object.__setattr__(self, "terms", tuple(terms))

    def evaluate(self, index: Index) -> Set[str]:
        if not self.terms:
            return set()
        postings_lists = [index.index.get(t, []) for t in self.terms]
        if any(not p for p in postings_lists):
            return set()

        doc_maps = [{p.doc_id: p.positions for p in plist} for plist in postings_lists]
        common_docs = set(doc_maps[0].keys())
        for dmap in doc_maps[1:]:
            common_docs.intersection_update(dmap.keys())

        matching_docs = set()
        for doc_id in common_docs:
            positions_list = [dmap[doc_id] for dmap in doc_maps]
            for p0 in positions_list[0]:
                if all((p0 + k) in positions_list[k] for k in range(1, len(positions_list))):
                    matching_docs.add(doc_id)
                    break
        return matching_docs

    def extract_terms(self) -> List[str]:
        return list(self.terms)


@dataclass(frozen=True)
class AndNode(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> Set[str]:
        return self.left.evaluate(index) & self.right.evaluate(index)

    def extract_terms(self) -> List[str]:
        return self.left.extract_terms() + self.right.extract_terms()


@dataclass(frozen=True)
class OrNode(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> Set[str]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def extract_terms(self) -> List[str]:
        return self.left.extract_terms() + self.right.extract_terms()


@dataclass(frozen=True)
class NotNode(QueryNode):
    node: QueryNode

    def evaluate(self, index: Index) -> Set[str]:
        all_docs = set(index.docs.keys())
        return all_docs - self.node.evaluate(index)

    def extract_terms(self) -> List[str]:
        return []


_TOKEN_RE = re.compile(r'\(|\)|"[^"]+"|[^\s()"]+')

class QueryParser:
    """Парсер рекурсивного спуску: OR -> AND (явний і неявний) -> NOT -> Primary."""

    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self) -> str:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> QueryNode | None:
        if not self.tokens:
            return None
        return self.parse_or()

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
        if tok == "(":
            self.consume()
            node = self.parse_or()
            if self.peek() == ")":
                self.consume()
            return node
        if tok and tok.startswith('"') and tok.endswith('"'):
            self.consume()
            phrase_terms = list(tokenize(tok[1:-1]))
            return PhraseNode(phrase_terms)

        raw = self.consume()
        terms = list(tokenize(raw))
        return TermNode(terms[0] if terms else raw.casefold())


def parse_query(query_str: str) -> QueryNode | None:
    raw_tokens = _TOKEN_RE.findall(query_str)
    return QueryParser(raw_tokens).parse()