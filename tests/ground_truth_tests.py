import pytest
import tempfile
import json
import os
import shutil
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from datetime import datetime
from rdflib.namespace import RDF, XSD
from src.ground_truth_generation.generate_ground_truth import (
    build_kg_for_article, POLIANN, BASE_ACTORS, 
    normalize_actor_name, create_uri, parse_date_expression
)
from scripts.run_ground_truth_generation import main

# =====================
# PATH FIXTURES
# =====================

@pytest.fixture
def data_paths():
    """Fixture providing paths to data directories"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    return {
        # Input directories
        'jsonl_data': os.path.join(project_root, "polianna-dataset/data/03a_processed_to_jsonl"),
        'json_data': os.path.join(project_root, "polianna-dataset/data/03b_processed_to_json"),
        
        # Output directory
        'output_dir': os.path.join(project_root, "polianna-processed/turtle"),
        
        # Test data directory
        'test_data': os.path.join(project_root, "test_data")
    }

# =====================
# FIXTURES
# =====================

@pytest.fixture
def article_graph(sample_article_dir):
    """Create the RDF graph once for all tests that need it"""
    g = build_kg_for_article(sample_article_dir)
    article_uri = list(g.subjects(RDF.type, POLIANN.PolicyArticle))[0]
    return g, article_uri

@pytest.fixture
def annotations_cache(sample_article_dir):
    """Cache all annotations to avoid repeated file I/O"""
    annotation_file = os.path.join(sample_article_dir, "Curated_Annotations.jsonl")
    
    if not os.path.exists(annotation_file):
        return []
    
    annotations = []
    with open(annotation_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    annotations.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip malformed lines but report them
                    print(f"Warning: Skipped malformed JSON line in {annotation_file}")
                
    return annotations

@pytest.fixture
def empty_article_dir():
    """Create a temporary directory with minimal article data"""
    temp_dir = tempfile.mkdtemp()
    
    # Create minimal policy_info.json
    policy_info = {
        "Titel": "EMPTY_Article",
        "CELEX_Number": "00000L0000"
    }
    with open(os.path.join(temp_dir, "policy_info.json"), "w", encoding="utf-8") as f:
        json.dump(policy_info, f)
    
    # Create minimal Raw_Text.txt
    with open(os.path.join(temp_dir, "Raw_Text.txt"), "w", encoding="utf-8") as f:
        f.write("Empty test article.")
    
    # Create empty Curated_Annotations.jsonl
    with open(os.path.join(temp_dir, "Curated_Annotations.jsonl"), "w", encoding="utf-8") as f:
        pass
    
    yield temp_dir
    
    # Clean up after the test
    import shutil
    shutil.rmtree(temp_dir)

# =====================
# HELPER FUNCTIONS
# =====================

def count_triples(g, subject, predicate, obj=None):
    """Count triples matching a pattern"""
    return len(list(g.triples((subject, predicate, obj))))

def filter_annotations(annotations, layer=None, feature=None, tag=None, exclude_unspecified=True):
    """Filter cached annotations based on criteria"""
    filtered = []
    
    for ann in annotations:
        # Check layer filter
        if layer and ann.get('layer') != layer:
            continue
            
        # Check feature filter
        if feature and ann.get('feature') != feature:
            continue
            
        # Check tag filter
        if tag and ann.get('tag') != tag:
            continue
        
        # Exclude unspecified tags if requested
        if exclude_unspecified and ann.get('tag') == "Unspecified":
            continue
            
        filtered.append(ann)
    
    return filtered

def count_filtered_annotations(annotations, layer=None, feature=None, tag=None, exclude_unspecified=True):
    """Count annotations based on filtering criteria"""
    filtered = filter_annotations(annotations, layer, feature, tag, exclude_unspecified)
    return len(filtered)

def count_recognized_actors(annotations, recognized_actors):
    """
    Count unique actors that would be recognized by the system.
    This simulates what happens in build_kg_for_article.
    """
    unique_actors = set()  # Use a set to track unique normalized actor names
    
    for annotation in annotations:
        text = annotation.get('text', '').strip().lower()
        normalized_name = normalize_actor_name(text)
        
        # Check if this actor would be recognized
        if normalized_name in recognized_actors:
            unique_actors.add(normalized_name)
    
    return len(unique_actors)

def get_all_triples_for_subject(g, subject):
    """Get all triples for a specific subject, useful for debugging"""
    return list(g.triples((subject, None, None)))

# =====================
# UNIT TESTS
# =====================

class TestUtilityFunctions:
    """Test the utility functions used in ground truth generation"""
    
    def test_create_uri(self):
        """Test URI creation function"""
        # Test basic URI creation
        uri = create_uri("TestEntity")
        assert isinstance(uri, URIRef), "Should create a URIRef"
        assert str(uri) == "https://polianna-kg.org/Ontology#TestEntity", "URI has incorrect format"
        
        # Test with numeric input
        uri = create_uri("32004L0008")
        assert str(uri) == "https://polianna-kg.org/Ontology#32004L0008", "URI has incorrect format for numeric input"
    
    def test_normalize_actor_name(self):
        """Test actor name normalization"""
        # Test basic normalization
        assert normalize_actor_name("member states") == "member states", "Basic normalization failed"
        
        # Test capitalization
        assert normalize_actor_name("Member States") == "member states", "Case normalization failed"
        
        # Test plural forms
        assert normalize_actor_name("member state") == "member states", "Plural/singular normalization failed"
        
        # Test possessives
        assert normalize_actor_name("member state's") == "member states", "Possessive normalization failed"
    
        # Test with non-breaking spaces
        assert normalize_actor_name("member\xa0states") == "member states", "Unicode space normalization failed"
        assert normalize_actor_name("member\\xa0states") == "member states", "Escaped space normalization failed"
        
        # Test synonyms
        assert normalize_actor_name("commission") == "european commission", "Synonym mapping failed"
        assert normalize_actor_name("european union") == "union", "Synonym mapping failed"
    
    def test_parse_date_expression(self):
        """Test date expression parsing"""
        # Test exact date formats
        date, parseable = parse_date_expression("21 February 2007")
        assert parseable, "Should recognize standard date format"
        assert date.year == 2007 and date.month == 2 and date.day == 21, "Incorrect date parsing"
        
        # Test year only
        date, parseable = parse_date_expression("2023")
        assert parseable, "Should recognize year-only format"
        assert date.year == 2023 and date.month == 1 and date.day == 1, "Incorrect year-only parsing"
        
        # Test unparseable date expressions
        date, parseable = parse_date_expression("within a reasonable time")
        assert not parseable, "Should not parse ambiguous time references"
        assert date is None, "Should return None for unparseable dates"
    
        date, parseable = parse_date_expression("within 18 months")
        assert not parseable, "Should not parse relative time expressions"
        assert date is None, "Should return None for unparseable dates"
        
        # Test empty string
        date, parseable = parse_date_expression("")
        assert not parseable, "Should handle empty strings gracefully"

# =====================
# GRAPH STRUCTURE TESTS
# =====================

class TestGraphStructure:
    """Test the structure of the generated RDF graph"""
    
    def test_basic_graph_structure(self, article_graph):
        """Test that the graph has the basic expected structure"""
        g, article_uri = article_graph
        
        # Test that classes and instance relationships are correct
        assert (article_uri, RDF.type, POLIANN.PolicyArticle) in g, "Article not typed as PolicyArticle"
        
        # Test article has full text
        assert any(g.triples((article_uri, POLIANN.fullText, None))), "Article missing fullText property"
        
        # Find the policy document
        policy_docs = list(g.subjects(RDF.type, POLIANN.PolicyDocument))
        assert len(policy_docs) == 1, "Expected exactly one PolicyDocument"
        
        # Test document-article relationship
        assert (policy_docs[0], POLIANN.hasArticle, article_uri) in g, "Document not linked to article correctly"
    
    def test_namespaces(self, article_graph):
        """Test that the required namespaces are defined"""
        g, _ = article_graph
        
        namespaces = {prefix: uri for prefix, uri in g.namespaces()}
        assert "pol" in namespaces, "POLIANN namespace not defined"
        assert str(namespaces["pol"]) == str(POLIANN), "POLIANN namespace incorrect"
        
        assert "eli" in namespaces, "ELI namespace not defined"
        assert "skos" in namespaces, "SKOS namespace not defined"
    
    def test_empty_article(self, empty_article_dir):
        """Test handling empty article with no annotations"""
        g = build_kg_for_article(empty_article_dir)
        
        # Should still create article and document nodes
        article_uris = list(g.subjects(RDF.type, POLIANN.PolicyArticle))
        assert len(article_uris) == 1, "Should create article node even for empty article"
        
        doc_uris = list(g.subjects(RDF.type, POLIANN.PolicyDocument))
        assert len(doc_uris) == 1, "Should create document node even for empty article"
        
        # Should have text but no annotations
        assert any(g.triples((article_uris[0], POLIANN.fullText, None))), "Should have fullText property"
        
        # Should not have any annotation-based triples
        assert count_triples(g, article_uris[0], POLIANN.contains_instrument, None) == 0, "Empty article should have no instruments"
        assert count_triples(g, article_uris[0], POLIANN.specifies_objective, None) == 0, "Empty article should have no objectives"

    def test_serialization_validity(self, article_graph):
        """Test that the graph can be serialized to valid TTL"""
        g, _ = article_graph
        
        # Write to a temporary TTL file
        with tempfile.NamedTemporaryFile(suffix='.ttl') as tmp:
            g.serialize(destination=tmp.name, format="turtle")
            
            # Try to parse it back to ensure it's valid
            try:
                test_g = Graph()
                test_g.parse(tmp.name, format="turtle")
                assert len(test_g) > 0, "Parsed graph is empty"
            except Exception as e:
                pytest.fail(f"Generated TTL has invalid syntax: {e}")

# =====================
# RELATIONSHIP TESTS
# =====================

@pytest.mark.parametrize("relation_type,predicate,layer,feature,tag", [
    ("instrument", POLIANN.contains_instrument, "Instrumenttypes", None, None),
    ("objective", POLIANN.specifies_objective, "Policydesigncharacteristics", "Objective", None),
    ("monitoring_form", POLIANN.specifies_monitoring_form, "Policydesigncharacteristics", "Compliance", "Form_monitoring"),
    ("sanctioning_form", POLIANN.specifies_sanctioning_form, "Policydesigncharacteristics", "Compliance", "Form_sanctioning"),
    ("compliance_time", POLIANN.specifies_compliance_time, "Policydesigncharacteristics", "Time", "Time_Compliance"),
    ("in_effect_time", POLIANN.specifies_in_effect_time, "Policydesigncharacteristics", "Time", "Time_InEffect"),
    ("monitoring_time", POLIANN.specifies_monitoring_time, "Policydesigncharacteristics", "Time", "Time_Monitoring"),
])
def test_relationship_counts(sample_article_dir, article_graph, annotations_cache, relation_type, predicate, layer, feature, tag):
    """Test that relationships are correctly extracted from annotations"""
    g, article_uri = article_graph
    
    # Get detailed debug information
    print(f"\n==== {relation_type.upper()} RELATIONSHIP COUNT DETAILS ====")
    
    # Count items in the graph
    actual_count = count_triples(g, article_uri, predicate, None)
    
    # Count expected items from annotations
    expected_count = count_filtered_annotations(annotations_cache, layer, feature, tag)
    
    # Print the detailed counts
    print(f"Expected count: {expected_count}")
    print(f"Actual count:   {actual_count}")
    
    # Show the filtered annotations details
    filtered_anns = filter_annotations(annotations_cache, layer, feature, tag)
    print(f"\nFound {len(filtered_anns)} matching annotations:")
    for idx, ann in enumerate(filtered_anns, 1):
        print(f"  {idx}. Text: '{ann.get('text')}' (Tag: {ann.get('tag')})")
    
    # Show the actual triples in the graph
    relationship_objects = list(g.objects(article_uri, predicate))
    print(f"\nFound {len(relationship_objects)} matching relationships in graph:")
    for idx, obj in enumerate(relationship_objects, 1):
        # Get object type and any annotations
        obj_type = next(g.objects(obj, RDF.type), "Unknown")
        obj_text = next(g.objects(obj, POLIANN.annotatedText), "No text")
        
        # Format the URI for better readability
        obj_str = str(obj)
        if "#" in obj_str:
            obj_str = obj_str.split("#")[1]
        
        print(f"  {idx}. URI: {obj_str}")
        print(f"     Type: {obj_type}")
        print(f"     Text: '{obj_text}'")
    
    # Make the assertion
    assert actual_count == expected_count, \
        f"Expected {expected_count} {relation_type}s, got {actual_count}"

@pytest.mark.parametrize("relation_type,predicate,tag", [
    ("establishes_authority", POLIANN.establishes_authority, "Authority_established"),
    ("grants_legislative_authority", POLIANN.grants_legislative_authority, "Authority_legislative"),
    ("grants_monitoring_authority", POLIANN.grants_monitoring_authority, "Authority_monitoring"),
    ("imposes_monitoring", POLIANN.imposes_monitoring, "Addressee_monitored"),
    ("grants_resources", POLIANN.grants_resources, "Addressee_resource"),
])
def test_specialized_actor_relations(sample_article_dir, article_graph, annotations_cache, relation_type, predicate, tag):
    """Test that specialized actor relationships are correctly extracted from annotations"""
    g, article_uri = article_graph
    
    print(f"\n==== SPECIALIZED ACTOR RELATIONSHIP: {relation_type.upper()} ====")
    
    # Count items in the graph
    actual_count = count_triples(g, article_uri, predicate, None)
    
    # Get filtered actor annotations
    actor_annotations = filter_annotations(
        annotations_cache,
        "Policydesigncharacteristics", 
        "Actor", 
        tag
    )
    
    # Print details about these specialized annotations
    print(f"\nFound {len(actor_annotations)} '{tag}' annotations:")
    for idx, ann in enumerate(actor_annotations, 1):
        text = ann.get('text', '').strip()
        normalized = normalize_actor_name(text.lower())
        is_recognized = normalized in BASE_ACTORS
        print(f"  {idx}. Original: '{text}'")
        print(f"     Normalized: '{normalized}' ({'recognized' if is_recognized else 'NOT recognized'})")
    
    # Show the actual triples in the graph for this relation
    relation_objects = list(g.objects(article_uri, predicate))
    print(f"\nFound {len(relation_objects)} '{relation_type}' relationships:")
    for idx, actor in enumerate(relation_objects, 1):
        # Format the actor URI for better readability
        actor_str = str(actor)
        if "http" in actor_str:
            if "#" in actor_str:
                actor_str = actor_str.split("#")[1]
            elif "/" in actor_str:
                actor_str = actor_str.split("/")[-1]
        
        print(f"  {idx}. Actor: {actor_str}")
    
    # Count only those that would be recognized by BASE_ACTORS
    expected_count = count_recognized_actors(actor_annotations, BASE_ACTORS)
    
    print(f"\nExpected recognized {relation_type}: {expected_count}")
    print(f"Actual {relation_type} count:        {actual_count}")
    
    assert actual_count == expected_count, \
        f"Expected {expected_count} {relation_type} relations, got {actual_count}"

def test_actor_relation_counts(sample_article_dir, article_graph, annotations_cache):
    """Test that actor relationships are correctly extracted"""
    g, article_uri = article_graph
    
    print("\n==== ACTOR RELATIONSHIP COUNT DETAILS ====")
    
    # Count different actor relations in the graph
    addresses_count = count_triples(g, article_uri, POLIANN.addresses, None)
    authorises_count = count_triples(g, article_uri, POLIANN.authorises, None)
    
    # Print summary of actor counts
    print(f"Addresses count:  {addresses_count}")
    print(f"Authorises count: {authorises_count}")
    
    # Get filtered actor annotations
    addressee_annotations = filter_annotations(
        annotations_cache,
        "Policydesigncharacteristics", 
        "Actor", 
        "Addressee_default"
    )
    
    authority_annotations = filter_annotations(
        annotations_cache,
        "Policydesigncharacteristics", 
        "Actor", 
        "Authority_default"
    )
    
    # Print details about addressee annotations
    print(f"\nAddressees - Found {len(addressee_annotations)} annotations:")
    for idx, ann in enumerate(addressee_annotations, 1):
        text = ann.get('text', '').strip()
        normalized = normalize_actor_name(text.lower())
        is_recognized = normalized in BASE_ACTORS
        print(f"  {idx}. Original: '{text}'")
        print(f"     Normalized: '{normalized}' ({'recognized' if is_recognized else 'NOT recognized'})")
    
    # Print details about authority annotations
    print(f"\nAuthorities - Found {len(authority_annotations)} annotations:")
    for idx, ann in enumerate(authority_annotations, 1):
        text = ann.get('text', '').strip()
        normalized = normalize_actor_name(text.lower())
        is_recognized = normalized in BASE_ACTORS
        print(f"  {idx}. Original: '{text}'")
        print(f"     Normalized: '{normalized}' ({'recognized' if is_recognized else 'NOT recognized'})")
    
    # Show the actual actor triples in the graph
    print("\nActor triples in graph:")
    
    # Addressee triples
    addressee_objects = list(g.objects(article_uri, POLIANN.addresses))
    print(f"\nFound {len(addressee_objects)} addresses relationships:")
    for idx, actor in enumerate(addressee_objects, 1):
        # Format the actor URI for better readability
        actor_str = str(actor)
        if "http" in actor_str:
            if "#" in actor_str:
                actor_str = actor_str.split("#")[1]
            elif "/" in actor_str:
                actor_str = actor_str.split("/")[-1]
        
        print(f"  {idx}. Actor: {actor_str}")
    
    # Authority triples
    authority_objects = list(g.objects(article_uri, POLIANN.authorises))
    print(f"\nFound {len(authority_objects)} authorises relationships:")
    for idx, actor in enumerate(authority_objects, 1):
        # Format the actor URI for better readability
        actor_str = str(actor)
        if "http" in actor_str:
            if "#" in actor_str:
                actor_str = actor_str.split("#")[1]
            elif "/" in actor_str:
                actor_str = actor_str.split("/")[-1]
        
        print(f"  {idx}. Actor: {actor_str}")
    
    # Count only those that would be recognized by BASE_ACTORS
    expected_addresses = count_recognized_actors(addressee_annotations, BASE_ACTORS)
    expected_authorises = count_recognized_actors(authority_annotations, BASE_ACTORS)
    
    print(f"\nExpected recognized addresses: {expected_addresses}")
    print(f"Actual addresses count:        {addresses_count}")
    print(f"Expected recognized authorises: {expected_authorises}")
    print(f"Actual authorises count:        {authorises_count}")
    
    assert addresses_count == expected_addresses, \
        f"Expected {expected_addresses} addresses relations, got {addresses_count}"
    assert authorises_count == expected_authorises, \
        f"Expected {expected_authorises} authorises relations, got {authorises_count}"

def test_ignored_features(sample_article_dir, article_graph, annotations_cache):
    """Test that features that should be ignored are properly skipped"""
    g, article_uri = article_graph
    
    print("\n==== IGNORED FEATURES TEST ====")
    
    # 1. Count annotations that should be skipped
    ignored_annotations = {
        'technology': filter_annotations(annotations_cache, layer="Technologyandapplicationspecificity"),
        'unspecified': filter_annotations(annotations_cache, tag="Unspecified", exclude_unspecified=False),
        'ignored_time': filter_annotations(annotations_cache, feature="Time", tag="Time_PolDuration") + 
                        filter_annotations(annotations_cache, feature="Time", tag="Time_Resources"),
        'sector': filter_annotations(annotations_cache, feature="Actor", tag="Adressee_sector"),
        'reference': filter_annotations(annotations_cache, feature="Reference"),
        'resource': filter_annotations(annotations_cache, feature="Resource")
    }
    
    # 2. Verify technology-specific features don't appear in the graph
    tech_predicates = [
        POLIANN.specifies_technology, POLIANN.specifies_energy, POLIANN.specifies_application
    ]
    
    for predicate in tech_predicates:
        tech_count = count_triples(g, article_uri, predicate, None)
        assert tech_count == 0, f"Technology predicate {predicate} should be ignored"
    
    # 3. Verify Unspecified tags are ignored
    # InstrumentType tags with "Unspecified" should be ignored
    instruments = list(g.objects(article_uri, POLIANN.contains_instrument))
    instrument_texts = []
    for instrument in instruments:
        # Instrument types don't have annotatedText in the graph, so we need to identify them by URI
        instrument_name = str(instrument).split('#')[-1]
        instrument_texts.append(instrument_name)
    
    assert "Unspecified" not in instrument_texts, "Unspecified tag should be ignored"
    
    # 4. Verify ignored Time tags
    time_nodes = []
    for p in [POLIANN.specifies_compliance_time, POLIANN.specifies_in_effect_time, POLIANN.specifies_monitoring_time]:
        time_nodes.extend(list(g.objects(article_uri, p)))

    ignored_time_tags = ["Time_PolDuration", "Time_Resources"]

    # Get all time node types in the graph
    time_node_types = set()
    for node in time_nodes:
        node_type = next(g.objects(node, RDF.type), None)
        if node_type:
            type_str = str(node_type).split('#')[-1] if '#' in str(node_type) else str(node_type)
            time_node_types.add(type_str)

    # Check that none of the ignored time tags appear as types in the graph
    for ignored_tag in ignored_time_tags:
        assert ignored_tag not in time_node_types, f"Ignored time tag '{ignored_tag}' should not appear as a node type"

    # Also verify that none of the predicates specific to ignored tags exist
    ignored_time_predicates = [
        POLIANN.specifies_policy_duration_time,  # If you have a predicate for Time_PolDuration
        POLIANN.specifies_resource_time          # If you have a predicate for Time_Resources
    ]

    for predicate in ignored_time_predicates:
        assert count_triples(g, article_uri, predicate, None) == 0, f"Predicate {predicate} for ignored time tag should not be used"
    
    # 5. Verify Addressee_sector is ignored
    actor_predicates = [
        POLIANN.addresses, POLIANN.authorises, POLIANN.grants_resources, 
        POLIANN.imposes_monitoring, POLIANN.establishes_authority,
        POLIANN.grants_legislative_authority, POLIANN.grants_monitoring_authority
    ]
    
    # Get all actor texts from the annotations that should be ignored
    sector_texts = [ann.get('text') for ann in ignored_annotations['sector']]
    
    # Check that none of the actor predicates reference actors from ignored annotations
    for predicate in actor_predicates:
        found_actors = list(g.objects(article_uri, predicate))
        for actor in found_actors:
            actor_uri = str(actor)
            # We don't have direct annotatedText for actors, but we can confirm
            # there's no sectors URI in the graph
            assert "sector" not in actor_uri.lower(), f"Sector actor should not appear in graph"
    
    # 6. Verify Reference and Resource features are ignored
    reference_texts = [ann.get('text') for ann in ignored_annotations['reference']]
    resource_texts = [ann.get('text') for ann in ignored_annotations['resource']]
    
    
    # Check that no triples with Reference or Resource exist
    has_reference = any(g.triples((article_uri, POLIANN.references_policy, None)))
    has_resource = any(g.triples((article_uri, POLIANN.specifies_resource, None)))
    
    assert not has_reference, "References should be ignored"
    assert not has_resource, "Resources should be ignored"
    
    # 7. Summary
    print("\nSummary of ignored features:")
    print(f"- Technology annotations: {len(ignored_annotations['technology'])} - All ignored: {tech_count == 0}")
    print(f"- Unspecified tags: {len(ignored_annotations['unspecified'])} - All ignored: {'Unspecified' not in instrument_texts}")
    print(f"- Ignored time tags: {len(ignored_annotations['ignored_time'])} - All ignored: {all(t not in time_node_types for t in ignored_time_tags)}")
    print(f"- Sector tags: {len(ignored_annotations['sector'])} - All ignored: {not any('sector' in str(a).lower() for p in actor_predicates for a in g.objects(article_uri, p))}")
    print(f"- Reference tags: {len(ignored_annotations['reference'])} - All ignored: {not has_reference}")
    print(f"- Resource tags: {len(ignored_annotations['resource'])} - All ignored: {not has_resource}")

# =====================
# END-TO-END TESTS
# =====================

@pytest.mark.parametrize("data_source", ["programmatic", "real_files"])
def test_end_to_end_pipeline(data_paths, data_source):
    """
    Test the complete pipeline using either programmatic or real data files.
    
    Args:
        data_paths: Fixture providing paths to data directories
        data_source: String indicating which data source to use ("programmatic" or "real_files")
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir)
        os.makedirs(output_dir)
        
        # Set up test data based on source type
        if data_source == "programmatic":
            # Create a programmatic test article
            article_dir = os.path.join(input_dir, "TEST_Article")
            os.makedirs(article_dir)
            
            # Create policy_info.json
            with open(os.path.join(article_dir, "policy_info.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "Titel": "TEST_Pipeline_Article",
                    "CELEX_Number": "00000L9999"
                }, f)
            
            # Create Raw_Text.txt
            with open(os.path.join(article_dir, "Raw_Text.txt"), "w", encoding="utf-8") as f:
                f.write("End-to-end pipeline test. The Commission shall prepare a report.")
            
            # Create Curated_Annotations.jsonl with various types of annotations
            annotations = [
                {"layer": "Instrumenttypes", "feature": "InstrumentType", "tag": "Edu_Outreach", 
                 "text": "prepare a report", "start": 25, "stop": 40, "span_id": "TEST1"},
                {"layer": "Policydesigncharacteristics", "feature": "Actor", "tag": "Authority_default", 
                 "text": "Commission", "start": 21, "stop": 31, "span_id": "TEST2"},
                {"layer": "Policydesigncharacteristics", "feature": "Objective", "tag": "Objective_QualIntention", 
                 "text": "pipeline test", "start": 10, "stop": 23, "span_id": "TEST3"},
                {"layer": "Policydesigncharacteristics", "feature": "Actor", "tag": "Authority_monitoring", 
                 "text": "Commission", "start": 21, "stop": 31, "span_id": "TEST4"}
            ]
            
            with open(os.path.join(article_dir, "Curated_Annotations.jsonl"), "w", encoding="utf-8") as f:
                for ann in annotations:
                    f.write(json.dumps(ann) + "\n")
                    
            # Expected relationships to verify
            expected_relationships = {
                POLIANN.contains_instrument: 1,
                POLIANN.authorises: 1, 
                POLIANN.grants_monitoring_authority: 1,
                POLIANN.specifies_objective: 1
            }
                    
        elif data_source == "real_files":
            # Find an article folder to test
            source_dir = data_paths['jsonl_data'] if os.path.exists(data_paths['jsonl_data']) else data_paths['test_data']
            
            if not os.path.exists(source_dir):
                pytest.skip(f"Source directory not found: {source_dir}")
            
            article_folders = []
            
            # Try to use the actual data directory first
            if os.path.exists(data_paths['jsonl_data']):
                article_folders = [d for d in os.listdir(data_paths['jsonl_data']) 
                                  if os.path.isdir(os.path.join(data_paths['jsonl_data'], d))]
            
            # Fall back to test_data if actual data not available
            if not article_folders and os.path.exists(data_paths['test_data']):
                article_folders = [d for d in os.listdir(data_paths['test_data']) 
                                  if os.path.isdir(os.path.join(data_paths['test_data'], d)) and 
                                  "Article" in d]
            
            if not article_folders:
                pytest.skip("No article folders found in input directories")
                
            # Copy a real article to our test directory
            source_article = os.path.join(source_dir, article_folders[0])
            dest_article = os.path.join(input_dir, os.path.basename(source_article))
            os.makedirs(dest_article)
            
            for file_name in os.listdir(source_article):
                src = os.path.join(source_article, file_name)
                if os.path.isfile(src):
                    dst = os.path.join(dest_article, file_name)
                    shutil.copy2(src, dst)
                    
            # With real files, we don't have specific expectations, just general structure
            expected_relationships = {}
        
        # Run the pipeline
        main(input_dir, output_dir)
        
        # Verify output was created
        assert len(os.listdir(output_dir)) > 0, "No output files generated"
        ttl_files = [f for f in os.listdir(output_dir) if f.endswith(".ttl")]
        assert len(ttl_files) > 0, "No TTL files generated"
        
        # Parse the output and verify it's valid
        g = Graph()
        g.parse(os.path.join(output_dir, ttl_files[0]), format="turtle")
        assert len(g) > 0, "Generated TTL graph is empty"
        
        # Verify basic structure
        assert any(g.subjects(RDF.type, POLIANN.PolicyArticle)), "No Policy Article found in graph"
        assert any(g.subjects(RDF.type, POLIANN.PolicyDocument)), "No Policy Document found in graph"
        
        # If we have specific expectations (programmatic test), verify them
        if expected_relationships:
            article_uri = list(g.subjects(RDF.type, POLIANN.PolicyArticle))[0]
            for predicate, expected_count in expected_relationships.items():
                actual_count = count_triples(g, article_uri, predicate, None)
                assert actual_count == expected_count, f"Expected {expected_count} {predicate} relations, got {actual_count}"
            
        print(f"✓ End-to-end test with {data_source} data successful: {len(g)} triples generated")