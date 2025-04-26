from rdflib import Graph, Namespace, URIRef, BNode, Literal, RDF
import pandas as pd
import pyshacl  # For SHACL validation
import os
import shutil
from pathlib import Path
import re
import numpy as np
from collections import defaultdict

### Namespaces
POLIANN = Namespace("https://polianna-kg.org/Ontology#")
ELI     = Namespace("http://data.europa.eu/eli/ontology#")
SKOS    = Namespace("http://www.w3.org/2004/02/skos/core#")

# Define time-related predicates and types
TIME_SPEC_PREDICATES = {
    POLIANN.specifies_compliance_time,
    POLIANN.specifies_in_effect_time,
    POLIANN.specifies_monitoring_time
}
TIME_INSTANCE_TYPES = {
    POLIANN.Time_Compliance,
    POLIANN.Time_InEffect,
    POLIANN.Time_Monitoring
}
TIME_DATE_PROPERTIES = {
    POLIANN.Time_Compliance: POLIANN.Compliance_date,
    POLIANN.Time_InEffect: POLIANN.In_effect_date,
    # POLIANN.Time_Monitoring doesn't have a specific date property in the GT generation
}

# Predicate sets for categorization
CORE_PREDICATES = {RDF.type, POLIANN.hasArticle, POLIANN.fullText}
POLICY_CHAR_PREDICATES = {POLIANN.contains_instrument}
COMPLIANCE_CHAR_PREDICATES = {
    POLIANN.contains_objective,
    POLIANN.contains_monitoring_form,
    POLIANN.contains_sanctioning_form
}
ACTOR_PREDICATES = {
    POLIANN.addresses, POLIANN.grants_resources, POLIANN.imposes_monitoring,
    POLIANN.authorises, POLIANN.establishes_authority,
    POLIANN.grants_legislative_authority, POLIANN.grants_monitoring_authority
}
# Time predicates are handled separately by TIME_SPEC_PREDICATES

class KnowledgeGraphEvaluator:
    """Evaluates generated knowledge graphs against ground truth"""
    
    KEY_RELATIONSHIPS = [
        POLIANN.contains_instrument,
        POLIANN.contains_objective,
        POLIANN.contains_monitoring_form,
        POLIANN.contains_sanctioning_form, 
        POLIANN.addresses
    ]

    # Named individuals for policy features
    POLICY_FEATURES = {
        "objective": POLIANN.Objective,
        "monitoring_form": POLIANN.Form_monitoring,
        "sanctioning_form": POLIANN.Form_sanctioning
    }

    def __init__(self, ground_truth_path=None, ontology_path=None):
        self.ground_truth = Graph()
        self.ontology = Graph()
        if ground_truth_path:
            self.load_ground_truth(ground_truth_path)
        if ontology_path:
            self.load_ontology(ontology_path)
    
    def load_ground_truth(self, path):
        """Load ground truth from a file"""
        self.ground_truth.parse(path, format="turtle")
        return len(self.ground_truth)
    
    def load_ontology(self, path):
        """Load ontology from a file"""
        self.ontology.parse(path, format="turtle")
        return len(self.ontology)
    
    def evaluate(self, generated_graph):
        """Evaluate a generated graph against the ground truth"""
        comprehensive_results = self.evaluate_comprehensive(generated_graph, self.ground_truth)
        return {k: comprehensive_results[k] for k in ["precision", "recall", "f1_score", "true_positives", "false_positives", "false_negatives"]}

    def evaluate_file(self, file_path, ground_truth_graph=None):
        """
        Evaluate a RDF file directly, handling potential parsing errors.
        
        Args:
            file_path: Path to the RDF file to evaluate (TTL or JSON-LD)
            ground_truth_graph: Optional ground truth graph (uses self.ground_truth if None)
            
        Returns:
            Dictionary with evaluation results, including error info if parsing fails
        """
        if ground_truth_graph is None:
            ground_truth_graph = self.ground_truth
            
        # Determine format based on file extension
        file_format = "json-ld" if str(file_path).lower().endswith(('.jsonld', '.json')) else "turtle"
        
        # First clean the file based on format
        cleaned = False
        if file_format == "turtle":
            cleaned = self.clean_ttl_file(file_path)
        else:  # JSON-LD
            cleaned = self.clean_jsonld_file(file_path)
        
        try:
            # Attempt to parse the file
            generated_graph = Graph()
            generated_graph.parse(file_path, format=file_format)
            
            # If successful, evaluate comprehensively
            results = self.evaluate_comprehensive(generated_graph, ground_truth_graph)
            
            # Add file metadata
            results["file_path"] = str(file_path)
            results["file_size"] = os.path.getsize(file_path)
            results["triple_count"] = len(generated_graph)
            results["cleaned"] = cleaned
            results["format"] = file_format
            
            return results
            
        except Exception as e:
            # If parsing fails, return error info
            return {
                "is_valid": False,
                "syntax_errors": 1,
                "syntax_error_message": str(e),
                "file_path": str(file_path),
                "file_size": os.path.getsize(file_path),
                "triple_count": 0,
                "cleaned": cleaned,
                "format": file_format
            }

    def clean_jsonld_file(self, file_path):
        """Clean a JSON-LD file by removing common LLM output markers and other artifacts."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            original_content = content
            cleaned_content = content
            
            # Only apply "jsonld" marker removal if they actually exist
            if content.strip().startswith('jsonld') or content.strip().endswith('jsonld'):
                # Remove standalone "jsonld" markers at the beginning or end of the file
                cleaned_content = re.sub(r'^jsonld\s*', '', cleaned_content)
                cleaned_content = re.sub(r'\s*jsonld$', '', cleaned_content)
            
            # Only remove markdown code blocks if they exist
            if '```json' in cleaned_content or '```jsonld' in cleaned_content or '```' in cleaned_content:
                # Remove Markdown code block markers
                cleaned_content = re.sub(r'```(?:json|jsonld)?\s*', '', cleaned_content)
                cleaned_content = re.sub(r'```\s*$', '', cleaned_content)
                cleaned_content = re.sub(r'```\s*', '', cleaned_content)
            
            # Only if other cleanings occurred, also clean these artifacts
            if cleaned_content != original_content:
                # Remove common JSON-LD LLM output artifacts
                cleaned_content = re.sub(r'<\|.*?\|>', '', cleaned_content)  # Remove special tokens like <|end|>
                cleaned_content = re.sub(r'^\s*Output:\s*', '', cleaned_content)  # Remove "Output:" prefixes
                
                # Fix common JSON-LD errors
                if '"@voca:' in cleaned_content:
                    cleaned_content = cleaned_content.replace('"@voca:', '"@vocab":')
            
            # Only write the file if we actually changed something
            if cleaned_content != original_content:
                # Verify the cleaned content is valid JSON before saving
                try:
                    import json
                    json.loads(cleaned_content)  # This will throw if the JSON is invalid
                    
                    # If we reach here, the JSON is valid, so save it
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(cleaned_content)
                    
                    return True
                except json.JSONDecodeError:
                    # If the cleaned content isn't valid JSON, keep the original
                    print(f"  Warning: Cleaning made {os.path.basename(file_path)} invalid, keeping original")
                    return False
            
            return False  # No changes were needed
            
        except Exception as e:
            print(f"  Error cleaning {os.path.basename(file_path)}: {str(e)}")
            return False
    
    def clean_ttl_file(self, file_path):
        """Clean a TTL file by removing Markdown code block markers and other artifacts."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            original_content = content
            
            # Remove Markdown code block markers
            if '```turtle' in content or '```ttl' in content or '```' in content:
                content = re.sub(r'```(?:turtle|ttl)?\s*', '', content)
                content = re.sub(r'```\s*$', '', content)
                content = re.sub(r'```\s*', '', content)
                
                # Remove binary string prefixes (b'')
                content = re.sub(r"b''(\^b')?", '', content)
                
                # Remove common LLM output artifacts
                content = re.sub(r'<\|.*?\|>', '', content)  # Remove special tokens like <|end|>
                content = re.sub(r'^\s*Output:\s*', '', content)  # Remove "Output:" prefixes
                
                # Save the cleaned content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return True if content != original_content else False
            return False
        except Exception as e:
            print(f"  Error cleaning {os.path.basename(file_path)}: {str(e)}")
            return False

    def _extract_time_details(self, graph):
        """
        Extracts details of time characteristics instances from a graph.

        Returns:
            dict: Mapping article URI -> predicate -> list of {'instance': URI/BNode, 'text': Literal, 'date': Literal | None}
        """
        time_details = defaultdict(lambda: defaultdict(list))
        processed_instances = set() # Track instances already processed

        # Find all time specification triples first
        for s, p, o in graph.triples((None, None, None)):
            if p in TIME_SPEC_PREDICATES and isinstance(o, (URIRef, BNode)):
                article_uri = s
                time_instance = o
                time_predicate = p

                # Now find details for this specific time_instance
                instance_type = None
                annotated_text = None
                structured_date = None

                for _, type_p, type_o in graph.triples((time_instance, RDF.type, None)):
                    if type_o in TIME_INSTANCE_TYPES:
                        instance_type = type_o
                        break # Assume one primary time type per instance

                if not instance_type:
                    continue # Skip if no valid time type found

                for _, text_p, text_o in graph.triples((time_instance, POLIANN.annotatedText, None)):
                    if isinstance(text_o, Literal):
                        annotated_text = text_o
                        break # Assume one annotated text

                date_prop = TIME_DATE_PROPERTIES.get(instance_type)
                if date_prop:
                    for _, date_p, date_o in graph.triples((time_instance, date_prop, None)):
                        if isinstance(date_o, Literal):
                            structured_date = date_o
                            break # Assume one structured date

                # Store details if we have at least annotated text
                if annotated_text is not None:
                     # Check if we already added details for this instance via another predicate
                    is_new = True
                    for pred_list in time_details[article_uri].values():
                        for existing_detail in pred_list:
                            if existing_detail['instance'] == time_instance:
                                is_new = False
                                break
                        if not is_new:
                            break

                    if is_new:
                        time_details[article_uri][time_predicate].append({
                            'instance': time_instance,
                            'type': instance_type,
                            'text': annotated_text,
                            'date': structured_date
                        })
                        processed_instances.add(time_instance)

        return time_details

    def _get_time_related_triples(self, graph, time_details):
        """Gets all triples associated with the identified time instances."""
        triples = set()
        all_time_instances = set()

        # Collect all instance URIs/BNodes from the details structure
        for article_uri, predicates in time_details.items():
            for predicate, details_list in predicates.items():
                 # Add the linking triple
                for details in details_list:
                    instance = details['instance']
                    all_time_instances.add(instance)
                    triples.add((article_uri, predicate, instance))

        # Add all triples where the instance is the subject
        for instance in all_time_instances:
            for s, p, o in graph.triples((instance, None, None)):
                triples.add((s, p, o))

        return triples

    def _categorize_triple(self, triple):
        """Categorizes a triple based on its predicate and structure."""
        s, p, o = triple

        # Core Structure Checks (more specific than just predicate)
        if p == RDF.type and o in {POLIANN.PolicyDocument, POLIANN.PolicyArticle}:
            return "core"
        if p == POLIANN.hasArticle: # Assuming subject is PolicyDocument, object is PolicyArticle URI
             return "core"
        if p == POLIANN.fullText: # Assuming subject is PolicyArticle URI
             return "core"

        # Predicate-based categorization
        if p in POLICY_CHAR_PREDICATES:
            return "policy_char"
        if p in COMPLIANCE_CHAR_PREDICATES:
             # Specific check for objective object type
             if p == POLIANN.contains_objective and o != POLIANN.Objective:
                 return "other" # Or handle as potential error?
             # Specific check for monitoring form object type
             if p == POLIANN.contains_monitoring_form and o != POLIANN.Form_monitoring:
                 return "other"
             # Specific check for sanctioning form object type
             if p == POLIANN.contains_sanctioning_form and o != POLIANN.Form_sanctioning:
                 return "other"
             return "compliance_char"
        if p in ACTOR_PREDICATES:
            return "actor"
        if p in TIME_SPEC_PREDICATES:
             # This should ideally not happen here as time triples are pre-filtered,
             # but acts as a safeguard.
             return "time"

        # Default to 'other' if no category matches
        return "other"

    def _evaluate_triple_accuracy(self, gt_graph, gen_graph):
        """
        Compares triples, handling time characteristics semantically and categorizing others.
        """
        gt_triples_full = set(gt_graph)
        gen_triples_full = set(gen_graph)

        # 1. Extract detailed time characteristics from both graphs
        gt_time_details = self._extract_time_details(gt_graph)
        gen_time_details = self._extract_time_details(gen_graph)

        # 2. Identify all triples related to these time characteristics
        gt_time_triples = self._get_time_related_triples(gt_graph, gt_time_details)
        gen_time_triples = self._get_time_related_triples(gen_graph, gen_time_details)

        # 3. Separate non-time triples
        gt_triples_non_time = gt_triples_full - gt_time_triples
        gen_triples_non_time = gen_triples_full - gen_time_triples

        # 4. Categorize non-time triples
        categories = ["core", "policy_char", "compliance_char", "actor", "other"]
        gt_categorized = defaultdict(set)
        gen_categorized = defaultdict(set)

        for triple in gt_triples_non_time:
            category = self._categorize_triple(triple)
            gt_categorized[category].add(triple)

        for triple in gen_triples_non_time:
            category = self._categorize_triple(triple)
            gen_categorized[category].add(triple)

        # 5. Calculate TP, FP, FN for each non-time category
        metrics_by_category = {}
        tp_non_time_total = 0
        fp_non_time_total = 0
        fn_non_time_total = 0

        for category in categories:
            gt_set = gt_categorized[category]
            gen_set = gen_categorized[category]

            tp = len(gt_set.intersection(gen_set))
            fp = len(gen_set - gt_set)
            fn = len(gt_set - gen_set)

            precision, recall, f1 = self._compute_precision_recall_f1(tp, tp + fp, tp + fn)

            metrics_by_category[category] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": precision, "recall": recall, "f1_score": f1
            }
            tp_non_time_total += tp
            fp_non_time_total += fp
            fn_non_time_total += fn

        # 6. Custom comparison for time characteristics (semantic matching)
        tp_time = 0
        fp_time = 0
        fn_time = 0
        matched_gen_instances = set() # Track matched generated instances

        # Iterate through ground truth time characteristics
        for article_uri, gt_predicates in gt_time_details.items():
            for predicate, gt_details_list in gt_predicates.items():
                for gt_detail in gt_details_list:
                    gt_text = gt_detail['text']
                    gt_date = gt_detail['date']
                    gt_type = gt_detail['type']
                    found_match = False

                    # Search for a matching characteristic in the generated graph
                    if article_uri in gen_time_details and predicate in gen_time_details[article_uri]:
                        # Make a copy to allow removal during iteration
                        gen_potential_matches = list(gen_time_details[article_uri][predicate])

                        for gen_idx, gen_detail in enumerate(gen_potential_matches):
                            gen_instance = gen_detail['instance']
                            # Skip if already matched to a GT instance or type mismatch
                            if gen_instance in matched_gen_instances or gen_detail['type'] != gt_type:
                                continue

                            gen_text = gen_detail['text']
                            gen_date = gen_detail['date']

                            # Check for match: type must match, and either text (contains, case-insensitive, stripped) or date (exact) must match
                            text_match = False
                            if gt_text is not None and gen_text is not None:
                                gt_str = str(gt_text).strip().lower()
                                gen_str = str(gen_text).strip().lower()
                                text_match = gt_str in gen_str

                            date_match = (gt_date is not None and gen_date is not None and gt_date == gen_date)

                            if text_match or date_match:
                                tp_time += 1
                                found_match = True
                                matched_gen_instances.add(gen_instance)
                                # Find the original index in the actual list and remove
                                original_list = gen_time_details[article_uri][predicate]
                                for i, item in enumerate(original_list):
                                     if item['instance'] == gen_instance:
                                         original_list.pop(i)
                                         break
                                break # Found a match for this gt_detail

                    if not found_match:
                        fn_time += 1 # This GT characteristic was not found

        # Count remaining generated time characteristics as false positives
        for article_uri, gen_predicates in gen_time_details.items():
            for predicate, gen_details_list in gen_predicates.items():
                 for gen_detail in gen_details_list:
                     # Check if instance was ever matched (it shouldn't be if it's still here)
                     if gen_detail['instance'] not in matched_gen_instances:
                         fp_time += 1
                         # Add to matched set now to avoid counting duplicates if it appears under multiple predicates
                         matched_gen_instances.add(gen_detail['instance'])

        # Add time metrics to the category dictionary
        time_precision, time_recall, time_f1 = self._compute_precision_recall_f1(tp_time, tp_time + fp_time, tp_time + fn_time)
        metrics_by_category["time"] = {
             "tp": tp_time, "fp": fp_time, "fn": fn_time,
             "precision": time_precision, "recall": time_recall, "f1_score": time_f1
        }

        # 7. Combine overall results
        total_tp = tp_non_time_total + tp_time
        total_fp = fp_non_time_total + fp_time
        total_fn = fn_non_time_total + fn_time

        # --- Verification Assert ---
        # Verify that the sum of TPs/FPs/FNs from all categories equals the calculated total
        assert total_tp == sum(m['tp'] for m in metrics_by_category.values()), "Mismatch in total True Positives"
        assert total_fp == sum(m['fp'] for m in metrics_by_category.values()), "Mismatch in total False Positives"
        assert total_fn == sum(m['fn'] for m in metrics_by_category.values()), "Mismatch in total False Negatives"
        # --- End Verification ---

        # Calculate overall precision, recall, f1
        overall_precision, overall_recall, overall_f1 = self._compute_precision_recall_f1(total_tp, total_tp + total_fp, total_tp + total_fn)

        return {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1_score": overall_f1,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "details_by_category": metrics_by_category # Nested dictionary with category metrics
        }

    def evaluate_comprehensive(self, gt_graph, gen_graph, shacl_shape_graph=None):
        """
        Extended evaluation with detailed metrics including category breakdown.
        Args:
            gt_graph: The ground truth RDF graph for comparison.
            gen_graph: The RDF graph generated by the LLM.
            shacl_shape_graph: Optional SHACL shape graph for validation.
        Returns:
            A dictionary containing detailed evaluation metrics.
        """
        # Check if the generated graph is valid
        syntax_metrics = self._evaluate_syntax_correctness(gen_graph)

        if not syntax_metrics["is_valid"]:
            # If the graph is invalid, return only syntax metrics
            return syntax_metrics

        # Basic metrics including category breakdown
        basic_metrics = self._evaluate_triple_accuracy(gt_graph, gen_graph)

        # Extract category details and flatten them
        category_details = basic_metrics.pop("details_by_category", {}) # Remove details from basic_metrics
        flattened_category_metrics = {}
        for category, metrics in category_details.items():
            for metric_name, value in metrics.items():
                # Create keys like 'f1_score_core', 'tp_time', 'recall_actor'
                flattened_category_metrics[f"{metric_name}_{category}"] = value

        # Property coverage
        property_metrics = self._evaluate_property_coverage(gen_graph, gt_graph)

        # Class coverage
        class_metrics = self._evaluate_class_coverage(gen_graph, gt_graph)

        # Relationship-specific metrics (consider if these overlap with categories and need adjustment)
        # For now, keep them separate as they evaluate specific object sets for key relationships
        relationship_metrics = self._evaluate_relationships(gen_graph, gt_graph)

        # Entity recognition evaluation
        entity_metrics = self._evaluate_entity_recognition(gen_graph, gt_graph)

        # Ontological consistency evaluation
        consistency_metrics = self._evaluate_ontological_consistency(gen_graph, shacl_shape_graph)

        # Combine all metrics
        result = {
            **syntax_metrics,
            **basic_metrics, # Contains overall P, R, F1, TP, FP, FN
            **flattened_category_metrics, # Contains category-specific metrics
            **property_metrics,
            **class_metrics,
            # **relationship_metrics, # Maybe skip this if categories cover it? Or keep for specific focus. Let's keep for now.
            **entity_metrics,
            **consistency_metrics,
            "ground_truth_triple_count": len(gt_graph)
        }

        # Add relationship metrics separately to avoid key clashes if needed
        # Or prefix them if keeping
        for rel_uri, metrics in relationship_metrics.items():
             rel_name = str(rel_uri).split('#')[-1]
             for metric_name, value in metrics.items():
                 # Prefix relationship metrics to distinguish, e.g., 'rel_precision_contains_instrument'
                 result[f"rel_{metric_name}_{rel_name}"] = value


        return result

    def _compute_precision_recall_f1(self, true_positives, predicted_count, actual_count):
        """Compute precision, recall, and F1 score."""
        precision = true_positives / predicted_count if predicted_count else 0
        recall = true_positives / actual_count if actual_count else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        return precision, recall, f1_score

    def _evaluate_set_comparison(self, extract_fn, generated_graph, ground_truth_graph, metric_prefix=""):
        """Generic evaluation using set comparison.
        
        Args:
            extract_fn: Function that extracts elements from a graph
            generated_graph: The generated RDF graph
            ground_truth_graph: The ground truth RDF graph
            metric_prefix: Prefix for metric names in returned dict
        
        Returns:
            Dictionary with precision, recall, and F1 metrics
        """
        generated_elements = set(extract_fn(generated_graph))
        ground_truth_elements = set(extract_fn(ground_truth_graph))
        
        true_positives = generated_elements & ground_truth_elements
        false_positives = generated_elements - ground_truth_elements
        false_negatives = ground_truth_elements - generated_elements
        
        precision, recall, f1_score = self._compute_precision_recall_f1(
            len(true_positives),
            len(generated_elements),
            len(ground_truth_elements)
        )
        
        if metric_prefix:
            prefix = f"{metric_prefix}_"
        else:
            prefix = ""
        return {
            f"{prefix}precision": precision,
            f"{prefix}recall": recall,
            f"{prefix}f1_score": f1_score,
            f"{prefix}true_positives": len(true_positives),
            f"{prefix}false_positives": len(false_positives),
            f"{prefix}false_negatives": len(false_negatives)
        }

    def _extract_properties(self, graph):
        """Extract properties from a graph."""
        return {p for _, p, _ in graph}

    def _evaluate_property_coverage(self, generated_graph, ground_truth_graph):
        """Evaluate the diversity and coverage of properties."""
        property_metrics = self._evaluate_set_comparison(
            self._extract_properties,  # Use named function for property extraction
            generated_graph,
            ground_truth_graph,
            "property"
        )
        
        # Add the extra property diversity metric
        property_metrics["property_diversity"] = len(self._extract_properties(generated_graph))
        
        return property_metrics

    def _extract_classes(self, graph):
        """Extract classes from a graph."""
        return {o for _, _, o in graph if isinstance(o, URIRef)}

    def _evaluate_class_coverage(self, generated_graph, ground_truth_graph):
        """Evaluate the coverage of important classes."""
        return self._evaluate_set_comparison(
            self._extract_classes,  # Use named function for class extraction
            generated_graph,
            ground_truth_graph,
            "class"
        )

    def _evaluate_relationships(self, generated_graph, ground_truth_graph):
        """Evaluate precision and recall for key relationships."""
        def extract_relationships(graph, relationship):
            """Extract objects for a specific relationship."""
            return {o for s, p, o in graph if p == relationship}

        relationship_metrics = {}
        for relationship in self.KEY_RELATIONSHIPS:
            relationship_metrics[str(relationship)] = self._evaluate_set_comparison(
                lambda g: extract_relationships(g, relationship),
                generated_graph,
                ground_truth_graph,
                metric_prefix=str(relationship).split('#')[-1]  # Use shortened name for metrics
            )

        return relationship_metrics

    def _evaluate_ontological_consistency(self, generated_graph, ontology_graph=None):
        """
        Evaluate ontological consistency using SHACL validation.
        
        Args:
            generated_graph: The RDF graph generated by the LLM
            ontology_graph: The ontology graph with constraints (if None, uses self.ontology)
            
        Returns:
            A dictionary containing metrics about constraint violations
        """
        # Use provided ontology graph or default to self.ontology
        if ontology_graph is None:
            if len(self.ontology) == 0:
                return {"ontology_validation": "skipped - no ontology loaded"}
            ontology_graph = self.ontology
        
        try:
            # Run SHACL validation
            conforms, results_graph, results_text = pyshacl.validate(
                generated_graph,
                shacl_graph=ontology_graph,
                inference='rdfs',
                abort_on_first=False
            )
            
            # Extract validation results
            if conforms:
                violations = 0
                violation_types = {}
            else:
                # Parse results to count violations
                violations = len(results_graph.subjects(
                    URIRef("http://www.w3.org/ns/shacl#resultSeverity"),
                    None
                ))
                
                # Count violations by type
                violation_types = {}
                for s, p, o in results_graph.triples((None, URIRef("http://www.w3.org/ns/shacl#sourceShape"), None)):
                    shape_type = str(o).split("/")[-1]
                    if shape_type in violation_types:
                        violation_types[shape_type] += 1
                    else:
                        violation_types[shape_type] = 1
            
            # Simplified consistency score calculation
            total_triples = len(generated_graph)
            consistency_score = 1 - (violations / total_triples) if total_triples > 0 else 1.0
            
            return {
                "ontology_consistency_score": max(0, consistency_score),
                "constraint_violations": violations,
                "constraint_violations_by_type": violation_types,
                "ontology_conforms": conforms
            }
        
        except Exception as e:
            # Handle validation errors gracefully
            return {
                "ontology_validation_error": str(e),
                "ontology_consistency_score": 0.0
            }

    def _evaluate_entity_recognition(self, generated_graph, ground_truth_graph):
        """Evaluate the accuracy of entity recognition."""
        return self._evaluate_set_comparison(
            lambda g: {s for s, _, _ in g if isinstance(s, URIRef)},  # Extract entities
            generated_graph,
            ground_truth_graph,
            "entity"
        )

    def _evaluate_syntax_correctness(self, generated_graph):
        """
        Evaluate the syntax correctness of the generated graph.
        
        Args:
            generated_graph: The RDF graph generated by the LLM.
        
        Returns:
            A dictionary containing syntax correctness metrics.
        """
        try:
            # Attempt to serialize the graph to check for syntax errors
            generated_graph.serialize(format="turtle")
            return {
                "is_valid": True,
                "syntax_errors": 0
            }
        except Exception as e:
            # Capture syntax errors
            return {
                "is_valid": False,
                "syntax_errors": 1,
                "syntax_error_message": str(e)
            }


# Helper functions for analysis and validation
def collect_problematic_files(results_df, output_dir, reasons=None):
    """
    Collect problematic files for further analysis.
    
    Args:
        results_df: DataFrame with evaluation results
        output_dir: Directory to copy problematic files to
        reasons: Dict with reason names and filter functions, e.g.,
                {'invalid': lambda x: not x['is_valid'],
                 'low_f1': lambda x: x['is_valid'] and x['f1_score'] < 0.2}
                 
    Returns:
        DataFrame with information about collected files
    """
    if reasons is None:
        reasons = {
            'invalid': lambda x: not x['is_valid'] if 'is_valid' in x else False,
            'low_f1': lambda x: x['is_valid'] and x['f1_score'] < 0.2 if 'is_valid' in x and 'f1_score' in x else False,
            'empty_graph': lambda x: x['is_valid'] and x['triple_count'] < 3 if 'is_valid' in x and 'triple_count' in x else False,
            'high_false_positives': lambda x: x['is_valid'] and x['false_positives'] > x['true_positives']*2 
                                   if 'is_valid' in x and 'false_positives' in x and 'true_positives' in x else False,
            'missing_policy_features': lambda x: x['is_valid'] and x.get('objective_recall', 1.0) < 0.5
                                   if 'is_valid' in x else False
        }
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize collection records
    collected = []
    
    # Process each reason
    for reason_name, filter_fn in reasons.items():
        # Create subdirectory for this reason
        reason_dir = os.path.join(output_dir, reason_name)
        os.makedirs(reason_dir, exist_ok=True)
        
        # Filter files matching this reason
        filtered = results_df[results_df.apply(filter_fn, axis=1)]
        
        # Copy each file to the output directory
        for _, row in filtered.iterrows():
            if 'file_path' not in row:
                continue
                
            src_path = row['file_path']
            if not os.path.exists(src_path):
                continue
                
            # Create destination path with model and strategy info
            filename = os.path.basename(src_path)
            
            # Add metrics to filename for easier browsing
            if 'f1_score' in row and 'model' in row:
                # Format: original_name--model--f1_0.23.ttl
                base, ext = os.path.splitext(filename)
                metrics_str = f"{base}--{row.get('model', 'unknown')}--f1_{row.get('f1_score', 0):.2f}{ext}"
                dst_path = os.path.join(reason_dir, metrics_str)
            else:
                dst_path = os.path.join(reason_dir, filename)
            
            # Copy the file
            try:
                shutil.copy2(src_path, dst_path)
                
                # Record the collection
                collected.append({
                    'original_path': src_path,
                    'collected_path': dst_path,
                    'reason': reason_name,
                    'article_id': row.get('article_id', 'unknown'),
                    'model': row.get('model', 'unknown'),
                    'strategy': row.get('strategy', 'unknown'),
                    'f1_score': row.get('f1_score', 0),
                    'is_valid': row.get('is_valid', False),
                    'true_positives': row.get('true_positives', 0),
                    'false_positives': row.get('false_positives', 0),
                    'triple_count': row.get('triple_count', 0)
                })
            except Exception as e:
                print(f"Error copying {src_path}: {e}")
    
    # Create a summary of collected files
    if collected:
        summary_df = pd.DataFrame(collected)
        
        # Add comparison files for context
        for reason_name in reasons.keys():
            if reason_name in summary_df['reason'].values:
                # Create a comparison directory with ground truth files
                comparison_dir = os.path.join(output_dir, reason_name, "ground_truth")
                os.makedirs(comparison_dir, exist_ok=True)
                
                # For each article with problematic files, copy the ground truth
                for article_id in summary_df[summary_df['reason'] == reason_name]['article_id'].unique():
                    if article_id and article_id != 'unknown':
                        # Find the ground truth file path (modify as needed)
                        ground_truth_path = None
                        
                        # Try some common locations for ground truth files
                        potential_gt_paths = [
                            os.path.join(os.path.dirname(os.path.dirname(output_dir)), "test_data", article_id, f"{article_id}_no_fulltext.ttl"),
                            os.path.join(os.path.dirname(os.path.dirname(output_dir)), "test_data", article_id, f"{article_id}.ttl")
                        ]
                        
                        for path in potential_gt_paths:
                            if os.path.exists(path):
                                ground_truth_path = path
                                break
                                
                        if ground_truth_path:
                            # Copy to comparison directory
                            dst = os.path.join(comparison_dir, f"{article_id}_ground_truth.ttl")
                            try:
                                shutil.copy2(ground_truth_path, dst)
                            except Exception as e:
                                print(f"Error copying ground truth {ground_truth_path}: {e}")
        
        # Create metadata JSON for each reason directory
        for reason, group in summary_df.groupby('reason'):
            metadata = {
                'reason': reason,
                'description': {
                    'invalid': "Files with syntax errors that couldn't be parsed",
                    'low_f1': "Files with valid syntax but poor semantic accuracy (F1 < 0.2)",
                    'empty_graph': "Files with very few triples (less than 3)",
                    'high_false_positives': "Files with many incorrect statements (false positives > 2x true positives)",
                    'missing_policy_features': "Files that fail to identify key policy features"
                }.get(reason, "Collection of problematic files"),
                'count': len(group),
                'models_affected': group['model'].unique().tolist(),
                'strategies_affected': group['strategy'].unique().tolist(),
                'article_ids': group['article_id'].unique().tolist(),
                'avg_f1_score': group['f1_score'].mean() if 'f1_score' in group else None
            }
            
            # Save metadata
            metadata_path = os.path.join(output_dir, reason, "metadata.json")
            try:
                with open(metadata_path, 'w') as f:
                    import json
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                print(f"Error writing metadata for {reason}: {e}")
            
        return summary_df
    
    return pd.DataFrame()


def generate_evaluation_report(results_df, output_dir=None):
    """
    Generate a comprehensive evaluation report from results DataFrame.
    
    Args:
        results_df: DataFrame with evaluation results
        output_dir: Directory to save report files (if None, just returns statistics)
        include_visualizations: Whether to generate plots (ignored - plots now handled in Jupyter notebooks)
        
    Returns:
        Dictionary with summary statistics
    """
    # Ensure output directory exists if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data - ensure valid boolean flag
    if 'is_valid' not in results_df.columns:
        results_df['is_valid'] = True
    
    # Basic statistics
    stats = {
        'total_files': len(results_df),
        'valid_files': results_df['is_valid'].sum(),
        'invalid_files': len(results_df) - results_df['is_valid'].sum(),
        'valid_percentage': results_df['is_valid'].mean() * 100
    }
    
    # Add model and strategy breakdowns
    if 'model' in results_df.columns:
        model_stats = results_df.groupby('model')['is_valid'].agg(['count', 'sum', 'mean'])
        model_stats = model_stats.rename(columns={'count': 'total', 'sum': 'valid', 'mean': 'valid_rate'})
        model_stats['valid_rate'] = model_stats['valid_rate'] * 100  # Convert to percentage
        stats['model_stats'] = model_stats.to_dict(orient='index')
        
    if 'strategy' in results_df.columns:
        strategy_stats = results_df.groupby('strategy')['is_valid'].agg(['count', 'sum', 'mean'])
        strategy_stats = strategy_stats.rename(columns={'count': 'total', 'sum': 'valid', 'mean': 'valid_rate'})
        strategy_stats['valid_rate'] = strategy_stats['valid_rate'] * 100  # Convert to percentage
        stats['strategy_stats'] = strategy_stats.to_dict(orient='index')
    
    # Combined model+strategy stats
    if 'model' in results_df.columns and 'strategy' in results_df.columns:
        combined_stats = results_df.groupby(['model', 'strategy'])['is_valid'].agg(['count', 'sum', 'mean'])
        combined_stats = combined_stats.rename(columns={'count': 'total', 'sum': 'valid', 'mean': 'valid_rate'})
        combined_stats['valid_rate'] = combined_stats['valid_rate'] * 100  # Convert to percentage
        stats['model_strategy_stats'] = combined_stats.to_dict(orient='index')
    
    # Performance metrics for valid files
    valid_df = results_df[results_df['is_valid']]
    if len(valid_df) > 0 and 'f1_score' in valid_df.columns:
        performance_metrics = ['precision', 'recall', 'f1_score']
        
        # Calculate overall performance
        stats['overall_performance'] = {
            metric: valid_df[metric].mean() for metric in performance_metrics if metric in valid_df.columns
        }
        
        # Calculate performance by model
        if 'model' in valid_df.columns:
            model_performance = valid_df.groupby('model')[performance_metrics].mean()
            stats['model_performance'] = model_performance.to_dict(orient='index')
            
        # Calculate performance by strategy
        if 'strategy' in valid_df.columns:
            strategy_performance = valid_df.groupby('strategy')[performance_metrics].mean()
            stats['strategy_performance'] = strategy_performance.to_dict(orient='index')
            
        # Combined model+strategy performance
        if 'model' in valid_df.columns and 'strategy' in valid_df.columns:
            combined_performance = valid_df.groupby(['model', 'strategy'])[performance_metrics].mean()
            stats['model_strategy_performance'] = combined_performance.to_dict(orient='index')
    
    # Save statistics as CSV summary tables
    if output_dir:
        # Save as JSON
        import json
        with open(os.path.join(output_dir, 'evaluation_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save detailed tabular summaries
        if len(results_df) > 0:
            # 1. Overall summary by model and strategy
            if 'model' in results_df.columns and 'strategy' in results_df.columns:
                overall_summary = results_df.pivot_table(
                    index='model', 
                    columns='strategy',
                    values=['is_valid', 'f1_score', 'precision', 'recall', 'triple_count'],
                    aggfunc={
                        'is_valid': ['count', 'mean'], 
                        'f1_score': ['mean', 'std'], 
                        'precision': ['mean'], 
                        'recall': ['mean'],
                        'triple_count': ['mean']
                    }
                )
                # Flatten column multi-index
                overall_summary.columns = ['_'.join(col).strip() for col in overall_summary.columns.values]
                overall_summary = overall_summary.reset_index()
                overall_summary.to_csv(os.path.join(output_dir, 'overall_summary.csv'), index=False)
            
            # 2. Article level summary
            if 'article_id' in results_df.columns:
                article_summary = results_df.pivot_table(
                    index='article_id',
                    columns=['model', 'strategy'],
                    values=['is_valid', 'f1_score'],
                    aggfunc={
                        'is_valid': 'sum',
                        'f1_score': 'mean'
                    }
                )
                article_summary.to_csv(os.path.join(output_dir, 'article_summary.csv'))
            
            # 3. Detailed metrics for valid files
            valid_metrics = valid_df.describe().T
            valid_metrics.to_csv(os.path.join(output_dir, 'valid_files_metrics.csv'))
            
            # 4. Save invalid file details
            invalid_df = results_df[~results_df['is_valid']]
            if len(invalid_df) > 0:
                invalid_summary = invalid_df[['file_path', 'file_name', 'model', 'strategy', 'syntax_error_message']]
                invalid_summary.to_csv(os.path.join(output_dir, 'invalid_files.csv'), index=False)
            
            # 5. Raw data
            results_df.to_csv(os.path.join(output_dir, 'raw_evaluation_data.csv'), index=False)
    
    return stats