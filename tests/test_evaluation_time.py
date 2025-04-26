import pytest
from rdflib import Graph, Namespace, URIRef, BNode, Literal
from rdflib.namespace import RDF
from pathlib import Path

from src.evaluation.evaluation import KnowledgeGraphEvaluator, POLIANN


def build_time_graph(article_uri, predicate, inst_type, text, date=None):
    g = Graph()
    inst = BNode()
    g.add((article_uri, predicate, inst))
    g.add((inst, RDF.type, inst_type))
    g.add((inst, POLIANN.annotatedText, Literal(text)))
    if date is not None:
        # determine correct date property
        date_pred = KnowledgeGraphEvaluator()._extract_time_details.__globals__["TIME_DATE_PROPERTIES"][inst_type]
        g.add((inst, date_pred, Literal(date)))
    return g, inst


def test_extract_time_details_single_instance():
    evaluator = KnowledgeGraphEvaluator()
    article = URIRef("http://example.org/article1")
    predicate = list(KnowledgeGraphEvaluator.__dict__["_extract_time_details"].__globals__["TIME_SPEC_PREDICATES"])[0]
    inst_type = list(KnowledgeGraphEvaluator.__dict__["_extract_time_details"].__globals__["TIME_INSTANCE_TYPES"])[0]
    text_val = "2025-01-01"
    date_val = "2025-01-01"

    g, inst = build_time_graph(article, predicate, inst_type, text_val, date_val)
    time_details = evaluator._extract_time_details(g)

    assert article in time_details
    assert predicate in time_details[article]
    details = time_details[article][predicate]
    assert len(details) == 1
    detail = details[0]
    assert detail['instance'] == inst
    assert str(detail['text']) == text_val
    assert str(detail['date']) == date_val
    assert detail['type'] == inst_type


def test_get_time_related_triples():
    evaluator = KnowledgeGraphEvaluator()
    article = URIRef("http://example.org/article2")
    predicate = list(KnowledgeGraphEvaluator.__dict__["_extract_time_details"].__globals__["TIME_SPEC_PREDICATES"])[0]
    inst_type = list(KnowledgeGraphEvaluator.__dict__["_extract_time_details"].__globals__["TIME_INSTANCE_TYPES"])[0]
    text_val = "Benchmark period"
    date_val = None

    g, inst = build_time_graph(article, predicate, inst_type, text_val, date_val)
    time_details = evaluator._extract_time_details(g)
    triples = evaluator._get_time_related_triples(g, time_details)

    # linking triple
    assert (article, predicate, inst) in triples
    # subject triples for instance
    assert (inst, RDF.type, inst_type) in triples
    assert (inst, POLIANN.annotatedText, Literal(text_val)) in triples


def test_evaluate_triple_accuracy_exact_match():
    evaluator = KnowledgeGraphEvaluator()
    # Build identical GT and Gen graphs
    gt = Graph()
    gen = Graph()
    # non-time triple
    gt.add((URIRef('http://ex/a'), URIRef('http://ex/p'), URIRef('http://ex/b')))
    gen.add((URIRef('http://ex/a'), URIRef('http://ex/p'), URIRef('http://ex/b')))
    # time triple
    article = URIRef("http://example.org/article3")
    predicate = list(KnowledgeGraphEvaluator.__dict__["_extract_time_details"].__globals__["TIME_SPEC_PREDICATES"])[0]
    inst_type = list(KnowledgeGraphEvaluator.__dict__["_extract_time_details"].__globals__["TIME_INSTANCE_TYPES"])[0]
    text_val = "01/2025"
    date_val = "2025-01"
    time_g, inst = build_time_graph(article, predicate, inst_type, text_val, date_val)
    # merge time triples
    for t in time_g:
        gt.add(t)
        gen.add(t)
    metrics = evaluator._evaluate_triple_accuracy(gt, gen)
    assert metrics['precision'] == pytest.approx(1.0)
    assert metrics['recall'] == pytest.approx(1.0)
    assert metrics['f1_score'] == pytest.approx(1.0)
    assert metrics['false_positives'] == 0
    assert metrics['false_negatives'] == 0


def test_evaluate_triple_accuracy_text_contains():
    evaluator = KnowledgeGraphEvaluator()
    gt = Graph()
    gen = Graph()
    article = URIRef("http://example.org/article4")
    # Use predicates/types directly
    predicate = POLIANN.specifies_compliance_time
    inst_type = POLIANN.Time_Compliance
    # GT with short text, no date
    gt_graph, gt_inst = build_time_graph(article, predicate, inst_type, "Feb 2025", None)
    # Gen with longer text containing different wording for month
    gen_graph, gen_inst = build_time_graph(article, predicate, inst_type, "In February 2025 the period started", None)
    for t in gt_graph:
        gt.add(t)
    for t in gen_graph:
        gen.add(t)
    metrics = evaluator._evaluate_triple_accuracy(gt, gen)
    # Expect NO match because "Feb 2025" is not a direct substring of "In February 2025..."
    # This results in 1 FN (GT instance not found) and 1 FP (Gen instance not matched)
    assert metrics['true_positives'] == 0
    assert metrics['false_positives'] == 1
    assert metrics['false_negatives'] == 1
    assert metrics['f1_score'] == pytest.approx(0.0)


def test_evaluate_triple_accuracy_missing_gen():
    evaluator = KnowledgeGraphEvaluator()
    gt = Graph()
    gen = Graph()
    article = URIRef("http://example.org/article5")
    predicate = list(KnowledgeGraphEvaluator.__dict__["_extract_time_details"].__globals__["TIME_SPEC_PREDICATES"])[0]
    inst_type = list(KnowledgeGraphEvaluator.__dict__["_extract_time_details"].__globals__["TIME_INSTANCE_TYPES"])[0]
    # GT only
    gt_graph, _ = build_time_graph(article, predicate, inst_type, "Mar 2025", None)
    for t in gt_graph:
        gt.add(t)
    # gen is empty
    metrics = evaluator._evaluate_triple_accuracy(gt, gen)
    assert metrics['true_positives'] == 0
    assert metrics['false_positives'] == 0
    assert metrics['false_negatives'] > 0
    assert metrics['precision'] == 0
    assert metrics['recall'] == 0
    assert metrics['f1_score'] == 0


def test_extract_time_details_from_ground_truth_file():
    evaluator = KnowledgeGraphEvaluator()
    # Load the provided ground truth TTL
    root = Path(__file__).parent.parent
    gt_path = root / 'test_data' / 'EU_32009L0028_Title_0_Chapter_0_Section_0_Article_04' / 'EU_32009L0028_Title_0_Chapter_0_Section_0_Article_04_no_fulltext.ttl'
    gt_graph = Graph()
    gt_graph.parse(str(gt_path), format='turtle')

    time_details = evaluator._extract_time_details(gt_graph)
    # There should be one article URI key
    assert len(time_details) == 1
    article_uri = next(iter(time_details))
    # Check compliance and monitoring predicates explicitly
    comp_pred = POLIANN.specifies_compliance_time
    mon_pred = POLIANN.specifies_monitoring_time
    # Expect 5 compliance instances and 2 monitoring instances
    assert len(time_details[article_uri][comp_pred]) == 5
    assert len(time_details[article_uri][mon_pred]) == 2

    # Check that instances with date have a non-null 'date'
    for detail in time_details[article_uri][comp_pred]:
        if detail['text'] and detail['type'] != POLIANN.Time_Compliance:
            continue
        # First three have dates, last two are durations
        if detail['date'] is not None:
            assert isinstance(detail['date'], Literal)


def test_evaluate_comprehensive_ground_truth_self():
    evaluator = KnowledgeGraphEvaluator()
    root = Path(__file__).parent.parent
    gt_path = root / 'test_data' / 'EU_32009L0028_Title_0_Chapter_0_Section_0_Article_04' / 'EU_32009L0028_Title_0_Chapter_0_Section_0_Article_04_no_fulltext.ttl'
    gt_graph = Graph()
    gt_graph.parse(str(gt_path), format='turtle')
    # Evaluate ground truth against itself
    metrics = evaluator.evaluate_comprehensive(gt_graph, gt_graph)
    # Perfect precision/recall/F1 on self-comparison
    assert metrics['precision'] == pytest.approx(1.0)
    assert metrics['recall'] == pytest.approx(1.0)
    assert metrics['f1_score'] == pytest.approx(1.0)
    assert metrics['false_positives'] == 0
    assert metrics['false_negatives'] == 0


def test_evaluate_triple_accuracy_false_positive_time():
    evaluator = KnowledgeGraphEvaluator()
    gt = Graph()
    gen = Graph()
    article = URIRef("http://example.org/article_fp")
    predicate = list(KnowledgeGraphEvaluator.__dict__["_extract_time_details"].__globals__["TIME_SPEC_PREDICATES"])[0]
    inst_type = list(KnowledgeGraphEvaluator.__dict__["_extract_time_details"].__globals__["TIME_INSTANCE_TYPES"])[0]
    # Gen has one time instance, GT empty -> one false positive
    gen_graph, _ = build_time_graph(article, predicate, inst_type, "Jun 2025", None)
    for t in gen_graph:
        gen.add(t)
    metrics = evaluator._evaluate_triple_accuracy(gt, gen)
    assert metrics['true_positives'] == 0
    assert metrics['false_positives'] == 1
    assert metrics['false_negatives'] == 0
    assert metrics['precision'] == 0
    assert metrics['recall'] == 0


def test_evaluate_triple_accuracy_date_only_match():
    evaluator = KnowledgeGraphEvaluator()
    gt = Graph()
    gen = Graph()
    article = URIRef("http://example.org/article_date")
    predicate = list(KnowledgeGraphEvaluator.__dict__["_extract_time_details"].__globals__["TIME_SPEC_PREDICATES"])[0]
    inst_type = list(KnowledgeGraphEvaluator.__dict__["_extract_time_details"].__globals__["TIME_INSTANCE_TYPES"])[0]
    # GT with date and text
    gt_graph, gt_inst = build_time_graph(article, predicate, inst_type, "May 2025", "2025-05-01")
    # Gen with different text but same date literal
    gen_graph = Graph()
    inst2 = BNode()
    gen_graph.add((article, predicate, inst2))
    gen_graph.add((inst2, RDF.type, inst_type))
    gen_graph.add((inst2, POLIANN.annotatedText, Literal("First day of May")))
    gen_graph.add((inst2, KnowledgeGraphEvaluator()._extract_time_details.__globals__["TIME_DATE_PROPERTIES"][inst_type], Literal("2025-05-01")))
    for t in gt_graph:
        gt.add(t)
    for t in gen_graph:
        gen.add(t)
    metrics = evaluator._evaluate_triple_accuracy(gt, gen)
    assert metrics['true_positives'] == 1
    assert metrics['false_positives'] == 0
    assert metrics['false_negatives'] == 0


def test_evaluate_triple_accuracy_text_only_match():
    evaluator = KnowledgeGraphEvaluator()
    gt = Graph()
    gen = Graph()
    article = URIRef("http://example.org/article_text")
    # Use explicit types/predicates
    predicate = POLIANN.specifies_compliance_time
    inst_type = POLIANN.Time_Compliance
    # GT with text only
    gt_graph, _ = build_time_graph(article, predicate, inst_type, "July 2025 period", None) # Removed date
    # Gen with longer text containing GT text, no date
    gen_graph, _ = build_time_graph(article, predicate, inst_type, "From July 2025 period start", None)
    for t in gt_graph:
        gt.add(t)
    for t in gen_graph:
        gen.add(t)
    metrics = evaluator._evaluate_triple_accuracy(gt, gen)
    # Expect a match because the text "July 2025 period" is in "From July 2025 period start" (case-insensitive substring)
    # and the type/predicate match. Neither has a date.
    assert metrics['true_positives'] == 1
    assert metrics['false_positives'] == 0
    assert metrics['false_negatives'] == 0
    assert metrics['f1_score'] == pytest.approx(1.0)


def test_evaluate_triple_accuracy_multiple_instances():
    evaluator = KnowledgeGraphEvaluator()
    gt = Graph()
    gen = Graph()
    article = URIRef("http://example.org/article_multi")
    predicate = list(KnowledgeGraphEvaluator.__dict__["_extract_time_details"].__globals__["TIME_SPEC_PREDICATES"])[0]
    inst_type = list(KnowledgeGraphEvaluator.__dict__["_extract_time_details"].__globals__["TIME_INSTANCE_TYPES"])[0]
    # GT has two instances
    gt1_graph, _ = build_time_graph(article, predicate, inst_type, "Jan 2025", "2025-01-01")
    gt2_graph, _ = build_time_graph(article, predicate, inst_type, "Feb 2025", "2025-02-01")
    for t in gt1_graph: gt.add(t)
    for t in gt2_graph: gt.add(t)
    # Gen matches first, includes one extra
    gen1_graph, _ = build_time_graph(article, predicate, inst_type, "Jan 2025", "2025-01-01")
    gen2_graph, _ = build_time_graph(article, predicate, inst_type, "Mar 2025", "2025-03-01")
    for t in gen1_graph: gen.add(t)
    for t in gen2_graph: gen.add(t)
    metrics = evaluator._evaluate_triple_accuracy(gt, gen)
    # Expect: 1 TP, 1 FP (Mar), 1 FN (Feb)
    assert metrics['true_positives'] == 1
    assert metrics['false_positives'] == 1
    assert metrics['false_negatives'] == 1


def test_evaluate_triple_accuracy_type_mismatch():
    evaluator = KnowledgeGraphEvaluator()
    gt = Graph()
    gen = Graph()
    article = URIRef("http://example.org/article_type")
    # compliance vs monitoring
    comp_pred = POLIANN.specifies_compliance_time
    mon_pred = POLIANN.specifies_monitoring_time
    comp_type = POLIANN.Time_Compliance
    mon_type = POLIANN.Time_Monitoring
    # GT compliance instance
    gt_inst_graph, _ = build_time_graph(article, comp_pred, comp_type, "April 2025", None)
    # Gen monitoring instance with same text
    gen_inst_graph, _ = build_time_graph(article, mon_pred, mon_type, "April 2025", None)
    for t in gt_inst_graph: gt.add(t)
    for t in gen_inst_graph: gen.add(t)
    metrics = evaluator._evaluate_triple_accuracy(gt, gen)
    # No match due to predicate/type mismatch
    assert metrics['true_positives'] == 0
    assert metrics['false_positives'] == 1
    assert metrics['false_negatives'] == 1


def test_evaluate_triple_accuracy_non_time_mismatch():
    evaluator = KnowledgeGraphEvaluator()
    gt = Graph()
    gen = Graph()
    # non-time triple mismatch
    a, p = URIRef('http://ex/a'), URIRef('http://ex/p')
    gt.add((a, p, URIRef('http://ex/b')))
    gen.add((a, p, URIRef('http://ex/c')))
    metrics = evaluator._evaluate_triple_accuracy(gt, gen)
    assert metrics['true_positives'] == 0
    assert metrics['false_positives'] == 1
    assert metrics['false_negatives'] == 1
