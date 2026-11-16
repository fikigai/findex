from findex.index import DocMeta, Index, Posting
from findex.query import AndNode, NotNode, OrNode, PhraseNode, TermNode, parse_query
from findex.ranking import BM25Scorer


def test_query_parser_ast():
    ast = parse_query('python AND (async OR await) NOT java "event loop"')
    expected = AndNode(
        AndNode(
            AndNode(
                TermNode("python"),
                OrNode(TermNode("async"), TermNode("await")),
            ),
            NotNode(TermNode("java")),
        ),
        PhraseNode(["event", "loop"]),
    )
    assert ast == expected


def test_operator_overloading_ast():
    node = TermNode("python") & (TermNode("async") | ~TermNode("java"))
    assert isinstance(node, AndNode)


def test_bm25_sanity_checks():
    index = Index(
        index_dict={
            "common": [
                Posting("d1", 1, (0,)),
                Posting("d2", 1, (0,)),
                Posting("d3", 1, (0,)),
            ],
            "rare": [Posting("d1", 1, (1,))],
        },
        docs_dict={
            "d1": DocMeta("d1", 10),
            "d2": DocMeta("d2", 10),
            "d3": DocMeta("d3", 100),
        },
    )
    scorer = BM25Scorer()
    # 1. Рідкий термін важить більше за частий
    assert scorer.score(index, ["rare"], "d1", 10) > scorer.score(index, ["common"], "d1", 10)
    # 2. Короткий документ з одним входженням вище за довгий
    assert scorer.score(index, ["common"], "d2", 10) > scorer.score(index, ["common"], "d3", 100)