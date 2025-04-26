import os
import json
import rdflib
import re
from rdflib import Namespace, URIRef, BNode, Literal
from rdflib.namespace import RDF, RDFS, XSD
import re
import datetime
from dateutil import parser

### Namespaces
POLIANN = Namespace("https://polianna-kg.org/Ontology#")
ELI     = Namespace("http://data.europa.eu/eli/ontology#")
SKOS    = Namespace("http://www.w3.org/2004/02/skos/core#")

def create_uri(local_name):
    """
    Helper to build a URIRef from a local name.
    Example: create_uri("EU_32004L0008") -> <https://polianna-kg.org/Ontology#EU_32004L0008>
    """
    return URIRef(str(POLIANN) + local_name)

# Define base actors dictionary at module level
BASE_ACTORS = {
    # EU Institutions
    "member states": create_uri("Member_States"),
    "european commission": URIRef("http://eurovoc.europa.eu/4038"),
    "european parliament": URIRef("http://eurovoc.europa.eu/2243"),
    "council": URIRef("http://eurovoc.europa.eu/114"),
    "european council": URIRef("http://eurovoc.europa.eu/128"),
    "union": URIRef("http://eurovoc.europa.eu/4060"),
    "acer": URIRef("http://eurovoc.europa.eu/447917"),
    "eurostat": URIRef("http://eurovoc.europa.eu/2198")
}

def normalize_actor_name(name):
    """Normalize actor names by handling plurals, possessives, and special characters."""
    # Remove quotes and lowercase
    name = name.lower().strip("'\"")
    
    # Replace non-breaking spaces with regular spaces
    name = name.replace('\xa0', ' ')
    name = name.replace("\\xa0", " ")
    name = name.replace("'", "")
    
    # Remove possessive forms
    name = re.sub(r"'s$", '', name)
    
    # Special case mapping for synonyms
    synonym_map = {
        "commission": "european commission",
        "european union": "union",
        "member state": "member states",
        "european union energy star board": "euesb",
        "commission (eurostat)": "eurostat",
    }
    
    if name in synonym_map:
        return synonym_map[name]
    
    # Special case for "member states" - preserve plural
    if name == "member states":
        return "member states"
    
    # Handle plurals (with exceptions)
    plural_exceptions = {
        "authorities": "authority",
        "bodies": "body",
        "countries": "country",
        "undertakings": "undertaking",
        "parties": "party",
        "companies": "company",
        "enterprises": "enterprise",
        "communities": "community",
        "suppliers": "supplier",
        "consumers": "consumer",
        "customers": "customer",
        "producers": "producer",
        "operators": "operator",
        "manufacturers": "manufacturer",
        "distributors": "distributor",
        "centres": "centre",
        "centers": "center",
    }
    
    for plural, singular in plural_exceptions.items():
        if name.endswith(plural):
            return name[:-len(plural)] + singular
    
    # General plural rule (if ends with 's' and not in exceptions)
    if name.endswith('s') and not name.endswith('ss') and len(name) > 3:
        return name[:-1]
    
    return name

def parse_date_expression(text):
    """
    Attempts to parse a date expression into a structured format.
    Returns a tuple of (parsed_date, is_parseable)
    """
    # Clean the text
    text = text.lower().replace('\xa0', ' ').strip("'\"")
    
        # Try to handle simple year mentions
    if re.match(r'^(\d{4})$', text):
        year = int(text)
        return (datetime.date(year, 1, 1), True)
    
    # Handle "day/month/year" formats
    date_patterns = [
        # 31 December 2020
        r'(\d{1,2})[ ]?(january|february|march|april|may|june|july|august|september|october|november|december)[ ]?(\d{4})',
        # 1 January 2021
        r'(\d{1,2})[ ]?(january|february|march|april|may|june|july|august|september|october|november|december)[ ]?(\d{4})',
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            day, month, year = match.groups()
            month_map = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
                'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
            }
            try:
                return (datetime.date(int(year), month_map[month], int(day)), True)
            except ValueError:
                pass
    
    # Try with dateutil parser as a fallback but with stricter rules
    try:
        # Only use fuzzy parsing if text contains recognizable date components
        if any(month in text for month in ['january', 'february', 'march', 'april', 'may', 'june', 
                                         'july', 'august', 'september', 'october', 'november', 'december']):
            parsed_date = parser.parse(text, fuzzy=True)
            return (parsed_date.date(), True)
        else:
            # Try strict parsing without fuzzy mode for other cases
            parsed_date = parser.parse(text, fuzzy=False)
            return (parsed_date.date(), True)
    except:
        pass
        
    # Return None for unparseable expressions
    return (None, False)

def build_kg_for_article(article_dir):
    """
    Reads policy_info.json, Raw_Text.txt, and Curated_Annotations.jsonl
    from article_dir, constructs an rdflib.Graph, and returns it.
    """

    g = rdflib.Graph()
    
    # Bind prefixes for readable Turtle
    g.bind("pol", POLIANN)
    g.bind("eli", ELI)
    g.bind("skos", SKOS)
    
    # 1) Load policy_info.json
    policy_info_path = os.path.join(article_dir, "policy_info.json")
    with open(policy_info_path, "r", encoding="utf-8") as f:
        policy_info = json.load(f)
    
    article_title = policy_info["Titel"]  # e.g. "EU_32004L0008_Title_0_Chapter_0_Section_0_Article_05"
    celex_number  = policy_info["CELEX_Number"]  # e.g. "32004L0008"
    # ELI might be "http://data.europa.eu/eli/dir/2004/8/oj", if needed
    
    # 2) Create or reference the PolicyDocument for that CELEX
    doc_uri = create_uri(celex_number)   # e.g. :32004L0008
    g.add((doc_uri, RDF.type, POLIANN.PolicyDocument))
    
    # 3) Create the PolicyArticle individual
    article_uri = create_uri(article_title)  # e.g. :EU_32004L0008_Title_0_Chapter_0_Section_0_Article_05
    g.add((article_uri, RDF.type, POLIANN.PolicyArticle))
    
    # Link them: doc -> hasArticle -> article
    g.add((doc_uri, POLIANN.hasArticle, article_uri))
    
    # 4) Load Raw_Text.txt for :fullText
    raw_text_path = os.path.join(article_dir, "Raw_Text.txt")
    if os.path.exists(raw_text_path):
        with open(raw_text_path, "r", encoding="utf-8") as f:
            full_text = f.read().strip()
        g.add((article_uri, POLIANN.fullText, Literal(full_text, datatype=XSD.string)))
    
    # 5) Load curated annotations
    curated_ann_path = os.path.join(article_dir, "Curated_Annotations.jsonl")
    with open(curated_ann_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # A utility set to avoid duplicating references
    used_instruments = set()
    used_monitoring_forms = set()
    used_sanctioning_forms = set()
    used_objectives = set()
    
    # Keep a small counter for generating local URIs for new PDCharacteristic or new Agents
    # TODO: Check the behaviour of this local counter
    local_counter = 0
    
    # Dictionary to track actors we've already seen in this document
    known_actors = {}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        annotation = json.loads(line)
        
        layer   = annotation["layer"]
        feature = annotation["feature"]
        tag     = annotation["tag"]
        snippet = annotation["text"]  # The actual text snippet
        # start   = annotation["start"]
        # stop    = annotation["stop"]
        
        # 5a) Skip "Unspecified" in Instrument types
        if tag == "Unspecified":
            continue
        
        # =========================
        # Handle Instrument Types
        # =========================
        if layer == "Instrumenttypes" and feature == "InstrumentType":
            inst_uri = create_uri(tag)  # e.g. :Edu_Outreach
            if tag not in used_instruments:
                g.add((article_uri, POLIANN.contains_instrument, inst_uri))
                used_instruments.add(tag)
        
        # ========================================
        # Handle Policy Design Characteristics
        # ========================================
        elif layer == "Policydesigncharacteristics":
            
            # Sometimes we also have references (Ref_OtherPolicy, etc.)
            # or times (Time_Compliance), or actors (Addressee_default), etc.
            
            # We map feature+tag to the correct property or approach:
            
            if feature == "Time":
                if tag == "Time_PolDuration" or tag == "Time_Resources":
                    continue
                
                if tag == "Time_Compliance":
                    local_counter += 1
                    time_uri = create_uri(f"{article_title}_TimeCompliance_{local_counter}")
                    g.add((time_uri, RDF.type, POLIANN.Time_Compliance))
                    g.add((time_uri, POLIANN.annotatedText, Literal(snippet)))
                    
                    # Try to parse the date expression
                    parsed_date, is_parseable = parse_date_expression(snippet)
                    if is_parseable and parsed_date:
                        # Add structured date information
                        g.add((time_uri, POLIANN.Compliance_date, Literal(parsed_date.isoformat(), datatype=XSD.date)))
                    
                    g.add((article_uri, POLIANN.specifies_compliance_time, time_uri))
                
                elif tag == "Time_InEffect":
                    local_counter += 1
                    time_uri = create_uri(f"{article_title}_TimeInEffect_{local_counter}")
                    g.add((time_uri, RDF.type, POLIANN.Time_InEffect))
                    g.add((time_uri, POLIANN.annotatedText, Literal(snippet)))
                    
                    # Try to parse the date expression
                    parsed_date, is_parseable = parse_date_expression(snippet)
                    if is_parseable and parsed_date:
                        # Add structured date information
                        g.add((time_uri, POLIANN.In_effect_date, Literal(parsed_date.isoformat(), datatype=XSD.date)))
                    
                    g.add((article_uri, POLIANN.specifies_in_effect_time, time_uri))
                
                elif tag == "Time_Monitoring":
                    local_counter += 1
                    time_uri = create_uri(f"{article_title}_TimeMonitoring_{local_counter}")
                    g.add((time_uri, RDF.type, POLIANN.Time_Monitoring))
                    g.add((time_uri, POLIANN.annotatedText, Literal(snippet)))
                    
                    g.add((article_uri, POLIANN.specifies_monitoring_time, time_uri))
            
            if feature == "Compliance":
                if tag == "Form_monitoring":
                    # Treat monitoring form as a SKOS concept
                    monitoring_form_uri = create_uri("Form_monitoring")
                    if "Form_monitoring" not in used_monitoring_forms:
                        g.add((article_uri, POLIANN.contains_monitoring_form, monitoring_form_uri))
                        used_monitoring_forms.add("Form_monitoring")
                
                elif tag == "Form_sanctioning":
                    # Treat sanctioning form as a SKOS concept
                    sanctioning_form_uri = create_uri("Form_sanctioning")
                    if "Form_sanctioning" not in used_sanctioning_forms:
                        g.add((article_uri, POLIANN.contains_sanctioning_form, sanctioning_form_uri))
                        used_sanctioning_forms.add("Form_sanctioning")
            
            elif feature == "Actor":
                prop_map = {
                    "Addressee_default"    : POLIANN.addresses,
                    "Addressee_resource"   : POLIANN.grants_resources,
                    "Addressee_monitored"  : POLIANN.imposes_monitoring,
                    "Authority_default"    : POLIANN.authorises,
                    "Authority_established": POLIANN.establishes_authority,
                    "Authority_legislative": POLIANN.grants_legislative_authority,
                    "Authority_monitoring" : POLIANN.grants_monitoring_authority
                }
                
                if tag == "Adressee_sector":
                    continue

                if tag in prop_map:
                    the_property = prop_map[tag]
                    
                    snippet_text = snippet.strip()
                    normalized_name = normalize_actor_name(snippet_text.lower())
                    
                    if normalized_name in known_actors:
                        actor_uri = known_actors[normalized_name]
                    elif normalized_name in BASE_ACTORS:  # Use the module-level constant here
                        actor_uri = BASE_ACTORS[normalized_name]
                        known_actors[normalized_name] = actor_uri
                    else:
                        continue
                    
                    g.add((article_uri, the_property, actor_uri))
            
            elif feature == "Objective":
                # Treat objective as a SKOS concept
                objective_uri = create_uri("Objective")
                if "Objective" not in used_objectives:
                    g.add((article_uri, POLIANN.contains_objective, objective_uri))
                    used_objectives.add("Objective")
            
            elif feature == "Reference" or feature == "Resource":
                continue  # Skip these for now

    return g