import re
import json
import os
import logging  # Added for logging skipped triplets
from rdflib import Graph, Namespace, URIRef, Literal

# Setup basic logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# --- Constants from rebuild_and_compare_triplets.py ---
namespaces = {
    '': 'https://polianna-kg.org/Ontology#',
    'eli': 'http://data.europa.eu/eli/ontology#',
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'skos': 'http://www.w3.org/2004/02/skos/core#',
    'eurovoc': 'http://eurovoc.europa.eu/',
    'xsd': 'http://www.w3.org/2001/XMLSchema#',
}

# After defining namespaces dict, add explicit 'pol' prefix
namespaces['pol'] = namespaces['']

literal_props = {'fullText', 'annotatedText', 'Compliance_date', 'Monitoring_date', 'In_effect_date'}
date_props = {'Compliance_date', 'Monitoring_date', 'In_effect_date'}
string_props = {'fullText'}

base_ns_uri = namespaces['']
literal_prop_uris = {base_ns_uri + prop for prop in literal_props}
date_prop_uris = {base_ns_uri + prop for prop in date_props}
string_prop_uris = {base_ns_uri + prop for prop in string_props}

known_pol_terms = {
    'PolicyArticle', 'PolicyDocument', 'TimeCharacteristic',
    'Time_Compliance', 'Time_InEffect', 'Time_Monitoring',
    'Edu_Outreach', 'Form_monitoring', 'Form_sanctioning',
    'FrameworkPolicy', 'Member_States', 'Objective',
    'PoliannaComplianceScheme', 'PoliannaInstrumentScheme',
    'PublicInvestment', 'PublicInvt', 'RD_D', 'RegulatoryInstr',
    'Subsidies_Incentives', 'TaxIncentives', 'TradablePermit',
    'VoluntaryAgrmt'
}

# --- Helpers ---
def expand_name(local_name):
    if ':' in local_name:
        prefix, local = local_name.split(':', 1)
        base = namespaces.get(prefix)
        return base + local if base else local_name
    return namespaces[''] + local_name


def expand_simplified_name(name, base_article_id, is_predicate=False):
    # Handle potential full URIs provided by the LLM
    if name.startswith('http://') or name.startswith('https://'):
        return name
    if is_predicate and name.lower() == 'type':  # Case-insensitive check for 'type'
        return namespaces['rdf'] + 'type'
    if ':' in name:
        return expand_name(name)
    # Case-insensitive check for 'Article' as subject/object
    if not is_predicate and name.lower() == 'article':
        return namespaces[''] + base_article_id
    # Case-insensitive check against known terms
    for term in known_pol_terms:
        if name.lower() == term.lower():
            return namespaces[''] + term
    if not is_predicate:
        doc_match = re.fullmatch(r"(\d{4,5}[LR]\d{4})", name, re.IGNORECASE)  # Ignore case for doc ID
        if doc_match:
            core = doc_match.group(1).upper()  # Normalize to upper case
            return namespaces[''] + core
    if is_predicate:
        return namespaces[''] + name
    return namespaces[''] + base_article_id + '_' + name.replace(" ", "_")  # Replace spaces


# --- Reconstruction from raw triplets ---
def extract_json_content(llm_output):
    json_pattern = r"```(?:json)\s*([\s\S]*?)```"
    matches = re.findall(json_pattern, llm_output)
    if matches:
        content = matches[0].strip()
    else:
        code_pattern = r"```([\s\S]*?)```"
        m2 = re.findall(code_pattern, llm_output)
        content = m2[0].strip() if m2 else llm_output.strip()
        if not (content.startswith('[') and content.endswith(']')) and \
           not (content.startswith('{') and content.endswith('}')):
            logging.warning("Extracted content from generic code block doesn't look like JSON.")
            if (llm_output.strip().startswith('[') and llm_output.strip().endswith(']')) or \
               (llm_output.strip().startswith('{') and llm_output.strip().endswith('}')):
                content = llm_output.strip()
    return content.replace("b'", "").replace("'b'", "").strip()


def validate_raw_json_triplets(json_string):
    try:
        data = json.loads(json_string)
        if isinstance(data, dict):
            if len(data) == 1:
                potential_list = next(iter(data.values()))
                if isinstance(potential_list, list):
                    data = potential_list
                else:
                    return False, None, 'JSON is a dictionary, but the value is not a list.'
            else:
                return False, None, 'JSON is a dictionary with multiple keys, expected a list or single-key dict containing a list.'
        if not isinstance(data, list):
            return False, None, 'JSON structure is not a list (or a dict containing a single list).'
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                logging.warning(f"Item at index {i} is not a dictionary: {item}")
                continue
            lower_keys = {k.lower() for k in item.keys()}
            has_spo_keys = all(k in lower_keys for k in ['s', 'p', 'o'])
            has_subject_predicate_object_keys = all(k in lower_keys for k in ['subject', 'predicate', 'object'])
            if not (has_spo_keys or has_subject_predicate_object_keys):
                logging.warning(f"Triplet at index {i} missing required keys (case-insensitive): {item}")
                continue
        return True, data, None
    except json.JSONDecodeError as e:
        return False, None, f'JSON decoding failed: {e}'
    except Exception as e:
        return False, None, f'Unexpected error during validation: {e}'


def reconstruct_graph_from_raw(raw_triplets_list, article_id):
    g = Graph()
    for prefix, uri in namespaces.items():
        g.bind(prefix if prefix else 'pol', uri)
    if not isinstance(raw_triplets_list, list):
        logging.error("Input to reconstruct_graph_from_raw is not a list.")
        return None
    for i, t in enumerate(raw_triplets_list):
        if not isinstance(t, dict):
            logging.warning(f"Skipping item at index {i}: not a dictionary ({type(t)}).")
            continue
        keys_lower = {k.lower(): k for k in t.keys()}
        s_key, p_key, o_key = None, None, None
        if all(k in keys_lower for k in ['s', 'p', 'o']):
            s_key, p_key, o_key = keys_lower['s'], keys_lower['p'], keys_lower['o']
        elif all(k in keys_lower for k in ['subject', 'predicate', 'object']):
            s_key, p_key, o_key = keys_lower['subject'], keys_lower['predicate'], keys_lower['object']
        else:
            logging.warning(f"Skipping triplet at index {i}: Missing required keys (s/p/o or Subject/Predicate/Object, case-insensitive). Found keys: {list(t.keys())}")
            continue
        try:
            s_name = str(t[s_key]).strip()
            p_name = str(t[p_key]).strip()
            o_name = str(t[o_key]).strip()
        except KeyError as e:
            logging.warning(f"Skipping triplet at index {i}: Error accessing key {e}. Dict: {t}")
            continue
        except Exception as e:
            logging.warning(f"Skipping triplet at index {i}: Error processing values. Dict: {t}. Error: {e}")
            continue
        if not s_name or not p_name or not o_name:
            logging.warning(f"Skipping triplet at index {i}: Contains empty subject, predicate, or object after stripping. Original: {t}")
            continue
        try:
            s = URIRef(expand_simplified_name(s_name, article_id, is_predicate=False))
            p = URIRef(expand_simplified_name(p_name, article_id, is_predicate=True))
            p_uri = str(p)
            if p_uri in literal_prop_uris:
                if p_uri in date_prop_uris:
                    if re.match(r'^\d{4}-\d{2}-\d{2}$', o_name):
                        obj = Literal(o_name, datatype=URIRef(namespaces['xsd']+'date'))
                    else:
                        logging.warning(f"Potential date literal for predicate {p_uri} does not match YYYY-MM-DD format: '{o_name}'. Treating as plain literal.")
                        obj = Literal(o_name)
                elif p_uri in string_prop_uris:
                    obj = Literal(o_name, datatype=URIRef(namespaces['xsd']+'string'))
                else:
                    obj = Literal(o_name)
            else:
                obj = URIRef(expand_simplified_name(o_name, article_id, is_predicate=False))
            g.add((s, p, obj))
        except Exception as e:
            logging.error(f"Error processing triplet at index {i}: {t}. Error: {e}", exc_info=True)
            continue
    return g


def extract_turtle_content(llm_output):
    if isinstance(llm_output, bytes):
        try:
            llm_output = llm_output.decode('utf-8', errors='ignore')
        except Exception:
            llm_output = str(llm_output)
    llm_output = str(llm_output)
    llm_output = re.sub(r"b'([^']*)'", r"\1", llm_output)
    llm_output = re.sub(r'b"([^"]*)"', r"\1", llm_output)
    llm_output = llm_output.replace("b''^b'", "").replace("b''", "")
    turtle_pattern = r"```(?:turtle|ttl)\s*([\s\S]*?)```"
    matches = re.findall(turtle_pattern, llm_output, re.IGNORECASE)
    if matches:
        content = matches[0].strip()
        return content
    code_pattern = r"```([\s\S]*?)```"
    matches = re.findall(code_pattern, llm_output)
    if matches:
        content = matches[0].strip()
        if '@prefix' in content.lower() or '@base' in content.lower() or ' a ' in content or ';' in content or '.' in content.strip().split('\n')[-1]:
            return content
        else:
            logging.warning("Content found in generic code block doesn't strongly resemble Turtle.")
    cleaned_output = llm_output.strip()
    if '@prefix' in cleaned_output.lower() or '@base' in cleaned_output.lower() or ' a ' in cleaned_output or ';' in cleaned_output or (cleaned_output.count('\n') > 0 and '.' == cleaned_output.strip().split('\n')[-1].strip()):
        return cleaned_output
    logging.warning("Could not reliably extract Turtle content. Returning original output.")
    return llm_output.strip()


def extract_jsonld_content(llm_output):
    if isinstance(llm_output, bytes):
        try:
            llm_output = llm_output.decode('utf-8', errors='ignore')
        except Exception:
            llm_output = str(llm_output)
    llm_output = str(llm_output)
    llm_output = re.sub(r"b'([^']*)'", r"\1", llm_output)
    llm_output = re.sub(r'b"([^"]*)"', r"\1", llm_output)
    jsonld_pattern = r"```(?:jsonld|json-ld|json)\s*([\s\S]*?)```"
    matches = re.findall(jsonld_pattern, llm_output, re.IGNORECASE)
    if matches:
        return matches[0].strip()
    code_pattern = r"```([\s\S]*?)```"
    matches = re.findall(code_pattern, llm_output)
    if matches:
        content = matches[0].strip()
        if (content.startswith('[') and content.endswith(']')) or \
           (content.startswith('{') and content.endswith('}')):
            return content
        else:
            logging.warning("Content found in generic code block doesn't look like JSON.")
    cleaned_output = llm_output.strip()
    if (cleaned_output.startswith('[') and cleaned_output.endswith(']')) or \
       (cleaned_output.startswith('{') and cleaned_output.endswith('}')):
        return cleaned_output
    logging.warning("Could not reliably extract JSON-LD content. Returning original output.")
    return llm_output.strip()


def validate_turtle(turtle_string):
    try:
        cleaned_string = turtle_string.replace('\\n', '\n').replace('\\t', '\t')
        cleaned_string = re.sub(r"b'([^']*)'", r"\1", cleaned_string)
        cleaned_string = re.sub(r'b"([^"]*)"', r"\1", cleaned_string)
        cleaned_string = cleaned_string.replace("b''^b'", "").replace("b''", "")
        if "```" in cleaned_string:
            pattern = r"```(?:turtle|ttl)?[\s\n]*(.*?)```"
            matches = re.findall(pattern, cleaned_string, re.DOTALL | re.IGNORECASE)
            if matches:
                cleaned_string = matches[0].strip()
            else:
                cleaned_string = re.sub(r"```(?:turtle|ttl)?\s*", "", cleaned_string).strip()
                cleaned_string = re.sub(r"```\s*$", "", cleaned_string).strip()
        g = Graph()
        g.parse(data=cleaned_string, format="turtle")
        return True, g, None
    except Exception as e:
        return False, None, str(e)


def validate_jsonld(jsonld_string):
    try:
        json_data = json.loads(jsonld_string)
        g = Graph()
        g.parse(data=jsonld_string, format="json-ld")
        return True, g, None
    except json.JSONDecodeError as e:
        return False, None, f"Invalid JSON: {str(e)}"
    except Exception as e:
        return False, None, str(e)


def parse_output(llm_output, format_type='ttl'):
    graph = None; error = None; is_valid = False; extracted_content = None
    if not llm_output:
        return {"is_valid": False, "graph": None, "error": "LLM output is empty.", "extracted_content": None, "original_output": llm_output}
    try:
        if format_type.lower() == 'ttl':
            extracted_content = extract_turtle_content(llm_output)
            if extracted_content:
                from rdflib import Graph as _G
                g = _G(); g.parse(data=extracted_content, format='turtle'); graph = g; is_valid = True
            else:
                error = "No Turtle content found"
        elif format_type.lower() in ['json-ld', 'jsonld']:
            extracted_content = extract_jsonld_content(llm_output)
            if extracted_content:
                from rdflib import Graph as _G
                g = _G(); g.parse(data=extracted_content, format='json-ld'); graph = g; is_valid = True
            else:
                error = "No JSON-LD content found"
        elif format_type.lower() == 'raw':
            extracted_content = extract_json_content(llm_output)
            if not extracted_content:
                error = "No JSON content found in the output."
                graph = None
            else:
                is_valid, data, validation_error = validate_raw_json_triplets(extracted_content)
                if is_valid:
                    graph = data
                    error = None
                else:
                    graph = None
                    error = validation_error
        else:
            error = f"Unsupported format: {format_type}"
    except Exception as e:
        error = str(e)
    return {"is_valid": is_valid, "graph": graph, "error": error, "extracted_content": extracted_content, "original_output": llm_output}


def parse_turtle_from_llm_output(llm_output):
    return parse_output(llm_output, format_type='ttl')