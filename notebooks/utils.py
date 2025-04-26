import os
from rdflib import Graph
from rdflib.namespace import Namespace
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import sys

# Add the project root to Python's path
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from src.ground_truth_generation.generate_ground_truth import BASE_ACTORS

# Define paths
turtle_dir = os.path.join(project_root, "polianna-processed", "turtle")
raw_text_dir = os.path.join(project_root, "polianna-dataset", "data", "03a_processed_to_jsonl")

# Define namespaces
POLIANN = Namespace("https://polianna-kg.org/Ontology#")
ELI     = Namespace("http://data.europa.eu/eli/ontology#")
SKOS    = Namespace("http://www.w3.org/2004/02/skos/core#")

def load_all_turtle_files(directory):
    """Load all .ttl files from a directory and return a dictionary of graphs."""
    graphs = {}
    file_count = 0
    
    for filename in os.listdir(directory):
        if filename.endswith(".ttl"):
            file_path = os.path.join(directory, filename)
            article_id = filename[:-4]  # Remove .ttl extension
            
            g = Graph()
            g.parse(file_path, format="turtle")
            graphs[article_id] = g
            file_count += 1
            
            if file_count % 100 == 0:
                print(f"Loaded {file_count} files...")
    
    print(f"Loaded {file_count} Turtle files in total.")
    return graphs


def export_test_dataset(test_df, output_dir):
    """
    Export the selected test articles to a directory structure for LLM experiments.
    
    Creates a folder for each article containing:
    - The ground truth solution (.ttl)
    - A version of the ground truth without the fullText property (.ttl)
    - The policy info file (.json)
    - The raw text file (.txt)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Export the article list
    test_df.to_csv(os.path.join(output_dir, 'test_articles.csv'), index=False)
    
    # Process each article
    for _, row in test_df.iterrows():
        article_id = row['article_id']
        
        # Create a directory for this article
        article_dir = os.path.join(output_dir, article_id)
        os.makedirs(article_dir, exist_ok=True)
        
        # Define source paths
        source_ttl = os.path.join(turtle_dir, f"{article_id}.ttl")
        source_dir = os.path.join(raw_text_dir, article_id)
        source_raw = os.path.join(source_dir, "Raw_Text.txt")
        source_info = os.path.join(source_dir, "policy_info.json")
        
        # Define destination paths
        dest_ttl = os.path.join(article_dir, f"{article_id}.ttl")
        dest_ttl_no_fulltext = os.path.join(article_dir, f"{article_id}_no_fulltext.ttl")
        dest_raw = os.path.join(article_dir, "Raw_Text.txt")
        dest_info = os.path.join(article_dir, "policy_info.json")
        
        # Copy and process TTL files
        if os.path.exists(source_ttl):
            # Copy the original TTL file
            with open(source_ttl, 'r', encoding='utf-8') as src, open(dest_ttl, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
            
            # Create a version without fullText property
            try:
                # Load the graph
                g = Graph()
                g.parse(source_ttl, format="turtle")
                
                # Create a new graph without fullText property
                g_no_fulltext = Graph()
                # Copy all namespace bindings
                for prefix, namespace in g.namespaces():
                    g_no_fulltext.bind(prefix, namespace)
                
                # Copy all triples except those with fullText predicate
                for s, p, o in g:
                    if p != POLIANN.fullText:
                        g_no_fulltext.add((s, p, o))
                
                # Save the modified graph
                g_no_fulltext.serialize(destination=dest_ttl_no_fulltext, format="turtle")
                
            except Exception as e:
                print(f"Error creating no-fulltext version for {article_id}: {str(e)}")
        else:
            print(f"Warning: TTL file not found for {article_id}")
                
        # Copy raw text file
        if os.path.exists(source_raw):
            with open(source_raw, 'r', encoding='utf-8') as src, open(dest_raw, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        else:
            print(f"Warning: Raw text file not found for {article_id}")
                
        # Copy policy info file
        if os.path.exists(source_info):
            with open(source_info, 'r', encoding='utf-8') as src, open(dest_info, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        else:
            print(f"Warning: Policy info file not found for {article_id}")
                
    print(f"Exported {len(test_df)} test articles to {output_dir}")


def extract_article_metrics(graphs):
    """
    Extract comprehensive article metrics based on the POLIANNA ontology structure.
    
    Args:
        graphs: Dictionary mapping article IDs to RDF graphs
        
    Returns:
        pandas.DataFrame: DataFrame containing extracted metrics for each article
    """
    metrics = []
    
    # Get all base actors from the BASE_ACTORS dictionary
    base_actor_names = {str(uri).split('#')[-1].split('/')[-1]: uri 
                       for name, uri in BASE_ACTORS.items()}
    
    for article_id, g in graphs.items():
        # Get total triples
        total_triples = len(g)
        
        # Get article text length
        text_query = """
        PREFIX polianna: <https://polianna-kg.org/Ontology#>
        SELECT ?text WHERE {
            ?article polianna:fullText ?text .
        } LIMIT 1
        """
        text_results = list(g.query(text_query))
        text_length = len(text_results[0][0]) if text_results else 0
        
        # Calculate density metric (characters per triple)
        density_metric = text_length / total_triples if total_triples > 0 else 0
        
        # Analyze the key object properties from the ontology
        actor_relations = {
            'direct_addresses': sum(1 for _ in g.objects(None, POLIANN.addresses) 
                            if not any(g.triples((None, POLIANN.imposes_monitoring, _)))),
            'direct_authorises': sum(1 for _ in g.objects(None, POLIANN.authorises)
                            if not any(g.triples((None, POLIANN.establishes_authority, _)) or g.triples((None, POLIANN.grants_legislative_authority, _)) or g.triples((None, POLIANN.grants_monitoring_authority, _)))),
            'establishes_authority': sum(1 for _ in g.objects(None, POLIANN.establishes_authority)),
            'grants_legislative_authority': sum(1 for _ in g.objects(None, POLIANN.grants_legislative_authority)),
            'grants_monitoring_authority': sum(1 for _ in g.objects(None, POLIANN.grants_monitoring_authority)),
            'imposes_monitoring': sum(1 for _ in g.objects(None, POLIANN.imposes_monitoring)),
        }
        
        # Analyze the design and compliance features
        design_relations = {
            'contains_instrument': sum(1 for _ in g.objects(None, POLIANN.contains_instrument)),
            'specifies_compliance_time': sum(1 for _ in g.objects(None, POLIANN.specifies_compliance_time)),
            'specifies_in_effect_time': sum(1 for _ in g.objects(None, POLIANN.specifies_in_effect_time)),
            'specifies_monitoring_time': sum(1 for _ in g.objects(None, POLIANN.specifies_monitoring_time)),
        }
        
        # Add binary indicators for compliance features (these are singleton concepts)
        compliance_features = {
            'has_objective': 1 if any(g.objects(None, POLIANN.contains_objective)) else 0,
            'has_monitoring_form': 1 if any(g.objects(None, POLIANN.contains_monitoring_form)) else 0,
            'has_sanctioning_form': 1 if any(g.objects(None, POLIANN.contains_sanctioning_form)) else 0
        }
        
        # Count total actors and instruments based on relationships
        actor_count = sum(actor_relations.values())
        instrument_count = design_relations['contains_instrument']
        
        # Analyze specific instrument types (these have multiple variants)
        instrument_types = {}
        instrument_dummy = {}
        if instrument_count > 0:
            instrument_query = """
            PREFIX polianna: <https://polianna-kg.org/Ontology#>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            
            SELECT ?instrument ?label WHERE {
                ?article polianna:contains_instrument ?instrument .
                OPTIONAL { ?instrument skos:prefLabel ?label }
            }
            """
            for row in g.query(instrument_query):
                instrument_uri = str(row[0])
                label = str(row[1]) if row[1] else instrument_uri.split('#')[-1]
                instrument_types[label] = instrument_types.get(label, 0) + 1
                instrument_dummy[label] = 1
        
        # Check for specific base actors
        actor_presence = {}
        for actor_name, actor_uri in base_actor_names.items():
            # Check if this actor appears in any relation
            is_present = False
            for relation in [POLIANN.addresses, POLIANN.authorises, POLIANN.establishes_authority, 
                           POLIANN.grants_legislative_authority, POLIANN.grants_monitoring_authority, 
                           POLIANN.imposes_monitoring]:
                if any(g.triples((None, relation, actor_uri))):
                    is_present = True
                    break
            actor_presence[actor_name] = 1 if is_present else 0
        
        # Count time characteristic classes
        design_classes = {
            'compliance_time_count': design_relations['specifies_compliance_time'],
            'in_effect_time_count': design_relations['specifies_in_effect_time'],
            'monitoring_time_count': design_relations['specifies_monitoring_time'],
        }

        # Create a metrics record
        record = {
            'article_id': article_id,
            'total_triples': total_triples,
            'text_length': text_length,
            'density_metric': density_metric,
            
            # Aggregate counts
            'actor_count': actor_count,
            'instrument_count': instrument_count,
            
            # Binary compliance indicators (replacing redundant counts)
            **compliance_features,
            
            # Detailed relationship counts
            **{f'relation_{k}': v for k, v in actor_relations.items()},
            **{f'relation_{k}': v for k, v in design_relations.items()},
            
            # Actor presence indicators
            **{f'actor_{k}': v for k, v in actor_presence.items()},
            
            # Design class counts
            **{f'class_{k}': v for k, v in design_classes.items()},
        }
        
        # Add instrument dummy variables
        for inst_type in ["Edu_Outreach", "RD_D", "RegulatoryInstr", "FrameworkPolicy", 
                         "Subsidies_Incentives", "VoluntaryAgrmt", "TradablePermit", 
                         "PublicInvt", "TaxIncentives"]:
            key = f'has_instrument_{inst_type}'
            record[key] = instrument_dummy.get(inst_type, 0)
            
        metrics.append(record)
    
    # Convert metrics list to DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    # Extract additional info from article_id
    if len(metrics_df) > 0:
        # Calculate property diversity score (for ontology coverage)
        property_columns = [col for col in metrics_df.columns if col.startswith('relation_')]
        metrics_df['property_diversity'] = metrics_df[property_columns].sum(axis=1)
        
        # Calculate class coverage score
        class_columns = [col for col in metrics_df.columns if col.startswith('class_')]
        metrics_df['class_diversity'] = metrics_df[class_columns].sum(axis=1)
        
        # Calculate instrument diversity
        instrument_columns = [col for col in metrics_df.columns if col.startswith('has_instrument_')]
        metrics_df['instrument_diversity'] = metrics_df[instrument_columns].sum(axis=1)
        
        # Calculate compliance feature coverage
        compliance_columns = ['has_objective', 'has_monitoring_form', 'has_sanctioning_form']
        metrics_df['compliance_diversity'] = metrics_df[compliance_columns].sum(axis=1)
        
        # Extract policy document ID and article components from article_id
        metrics_df['policy_id'] = metrics_df['article_id'].apply(lambda x: x.split('_')[1])
        metrics_df['chapter'] = metrics_df['article_id'].apply(lambda x: x.split('_')[5])
        metrics_df['section'] = metrics_df['article_id'].apply(lambda x: x.split('_')[7])
        
        # Updated coverage score with compliance features
        metrics_df['coverage_score'] = (
            metrics_df['property_diversity'] + 
            metrics_df['class_diversity'] * 2 + 
            metrics_df['instrument_diversity'] * 2 +
            metrics_df['compliance_diversity'] * 2
        )
    
    return metrics_df


def rename_actor_columns(df):
    """
    Rename actor columns in the article metrics DataFrame to use proper labels from the ontology.
    
    Args:
        df: pandas DataFrame containing article metrics
        
    Returns:
        pandas DataFrame with renamed columns
    """
    # Create mapping from fragment identifiers to proper labels
    actor_mapping = {
        'actor_114': 'actor_Council_of_EU',          # Council of the European Union
        'actor_128': 'actor_European_Council',       # European Council
        'actor_2198': 'actor_Eurostat',              # Eurostat
        'actor_2243': 'actor_European_Parliament',   # European Parliament
        'actor_4038': 'actor_European_Commission',   # European Commission
        'actor_4060': 'actor_European_Union',
        'actor_447917': 'actor_ACER',                # European Union Agency for the Cooperation of Energy Regulators
        'actor_Member_States': 'actor_Member_States',# Member States (already properly named)
        'actor_count': 'count_actor'                 # Count of actors
    }
    
    # Create mapping for policy features (now as skos:Concept instances)
    feature_mapping = {
        'has_objective': 'has_policy_objective',
        'has_monitoring_form': 'has_monitoring_provisions',
        'has_sanctioning_form': 'has_sanctioning_provisions'
    }
    
    # Create a comprehensive dictionary with all mappings
    column_mapping = {**actor_mapping, **feature_mapping}
    
    # Create a copy of the DataFrame to avoid modifying the original
    renamed_df = df.copy()
    
    # Rename columns that exist in the DataFrame
    for old_name, new_name in column_mapping.items():
        if old_name in renamed_df.columns:
            renamed_df.rename(columns={old_name: new_name}, inplace=True)
            
    # Add friendly display names for instrument columns
    instrument_mapping = {
        'has_instrument_Edu_Outreach': 'has_instrument_Education_Outreach',
        'has_instrument_RD_D': 'has_instrument_Research_Development',
        'has_instrument_RegulatoryInstr': 'has_instrument_Regulatory',
        'has_instrument_FrameworkPolicy': 'has_instrument_Framework_Policy',
        'has_instrument_Subsidies_Incentives': 'has_instrument_Subsidies',
        'has_instrument_VoluntaryAgrmt': 'has_instrument_Voluntary_Agreement',
        'has_instrument_TradablePermit': 'has_instrument_Tradable_Permit',
        'has_instrument_PublicInvt': 'has_instrument_Public_Investment',
        'has_instrument_TaxIncentives': 'has_instrument_Tax_Incentives'
    }
    
    # Rename instrument columns
    for old_name, new_name in instrument_mapping.items():
        if old_name in renamed_df.columns:
            renamed_df.rename(columns={old_name: new_name}, inplace=True)
    
    return renamed_df


def load_and_visualize_graph(article_id, turtle_dir=None, show_labels=True, figsize=(14, 10), 
                            graph_type="multidigraph", layout="spring", color_scheme="default",
                            node_size=800, font_size=10, max_nodes_to_label=100):
    """
    Load an RDF graph for a specified article ID and create an enhanced visualization.
    
    Args:
        article_id: The ID of the article
        turtle_dir: Directory containing the TTL files (defaults to global turtle_dir)
        show_labels: Whether to show node labels in the visualization
        figsize: Size of the visualization figure
        graph_type: Type of graph to create ('multidigraph', 'digraph', or 'graph')
        layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'shell', 'spectral')
        color_scheme: Color scheme for nodes/edges ('default', 'pastel', 'bright', 'dark')
        node_size: Size of nodes in the visualization
        font_size: Size of the node labels
        max_nodes_to_label: Maximum number of nodes to label (avoids cluttering)
        
    Returns:
        The loaded RDF graph and the NetworkX graph
    """
    import networkx as nx
    from rdflib.extras.external_graph_libs import (
        rdflib_to_networkx_multidigraph,
        rdflib_to_networkx_digraph,
        rdflib_to_networkx_graph
    )
    
    # Use default directory if none provided
    if turtle_dir is None:
        turtle_dir = os.path.join(project_root, "polianna-processed", "turtle")
    
    # Construct file path
    file_path = os.path.join(turtle_dir, f"{article_id}.ttl")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    # Load the graph
    g = Graph()
    g.parse(file_path, format="turtle")
    
    # Print the TTL content
    print(f"TTL Content for {article_id}:")
    print("-" * 80)
    ttl_content = g.serialize(format="turtle").decode("utf-8") if hasattr(g.serialize(format="turtle"), 'decode') else g.serialize(format="turtle")
    print(ttl_content)
    print("-" * 80)
    
    # Define color schemes
    color_schemes = {
        "default": {"node": "lightblue", "edge": "black", "label": "black", "bg": "white"},
        "pastel": {"node": "#b3e0ff", "edge": "#6699cc", "label": "#003366", "bg": "#f5f5f5"},
        "bright": {"node": "#ff9999", "edge": "#ff3333", "label": "#800000", "bg": "#ffffe6"},
        "dark": {"node": "#336699", "edge": "#003366", "label": "#e6e6e6", "bg": "#1a1a1a"}
    }
    colors = color_schemes.get(color_scheme, color_schemes["default"])
    
    # Convert RDF graph to NetworkX graph based on selected type
    if graph_type == "multidigraph":
        # Create multidigraph with predicate as the edge key
        nx_graph = rdflib_to_networkx_multidigraph(g, edge_attrs=lambda s, p, o: {"key": p, "label": str(p).split('/')[-1].split('#')[-1]})
    elif graph_type == "graph":
        # Create undirected graph with triples as attributes
        nx_graph = rdflib_to_networkx_graph(g)
    else:
        # Default to directed graph with triples as attributes
        nx_graph = rdflib_to_networkx_digraph(g)
    
    # Create simplified labels for nodes (last part of URI or literal value)
    node_labels = {}
    for node in nx_graph.nodes():
        if hasattr(node, 'toPython'):  # For literals
            label = str(node.toPython())
        else:
            label = str(node).split('/')[-1]
            if '#' in label:
                label = label.split('#')[-1]
        
        # Truncate long labels
        if len(label) > 25:
            label = label[:22] + "..."
            
        node_labels[node] = label
    
    # Create the visualization
    plt.figure(figsize=figsize, facecolor=colors["bg"])
    
    # Choose layout algorithm
    if layout == "circular":
        pos = nx.circular_layout(nx_graph)
    elif layout == "kamada_kawai":
        try:
            pos = nx.kamada_kawai_layout(nx_graph)
        except:
            print("Kamada-Kawai layout failed, falling back to spring layout")
            pos = nx.spring_layout(nx_graph, k=0.8, iterations=50, seed=42)
    elif layout == "shell":
        pos = nx.shell_layout(nx_graph)
    elif layout == "spectral":
        try:
            pos = nx.spectral_layout(nx_graph)
        except:
            print("Spectral layout failed, falling back to spring layout")
            pos = nx.spring_layout(nx_graph, k=0.8, iterations=50, seed=42)
    else:
        # Default to spring layout with better parameters
        pos = nx.spring_layout(nx_graph, k=0.8, iterations=50, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        nx_graph, pos, 
        node_size=node_size, 
        node_color=colors["node"],
        edgecolors='black',
        alpha=0.8
    )
    
    # Draw edges with arrows
    if graph_type in ["multidigraph", "digraph"]:
        nx.draw_networkx_edges(
            nx_graph, pos, 
            arrowsize=15, 
            width=1.5, 
            alpha=0.7,
            edge_color=colors["edge"],
            arrows=True,
            connectionstyle="arc3,rad=0.1"  # Curved edges
        )
    else:
        nx.draw_networkx_edges(
            nx_graph, pos, 
            width=1.5, 
            alpha=0.7,
            edge_color=colors["edge"]
        )
    
    # Draw node labels if requested and not too many nodes
    if show_labels and len(nx_graph.nodes) <= max_nodes_to_label:
        nx.draw_networkx_labels(
            nx_graph, pos, 
            labels=node_labels,
            font_size=font_size,
            font_color=colors["label"],
            font_weight='bold'
        )
    elif show_labels:
        print(f"Too many nodes ({len(nx_graph.nodes)}) to label clearly. Showing graph without labels.")
    
    # For multidigraph, draw edge labels (if not too cluttered)
    if graph_type == "multidigraph" and len(nx_graph.edges) <= 50:
        edge_labels = nx.get_edge_attributes(nx_graph, 'label')
        nx.draw_networkx_edge_labels(
            nx_graph, pos,
            edge_labels=edge_labels,
            font_size=font_size-2,
            font_color=colors["label"],
            alpha=0.7,
            rotate=False,
            label_pos=0.5
        )
    
    plt.title(f"RDF Graph for Article: {article_id}", fontsize=14, color=colors["label"])
    plt.axis("off")  # Hide axes
    plt.tight_layout()
    
    # Set background color
    ax = plt.gca()
    ax.set_facecolor(colors["bg"])
    
    plt.show()
    
    # Print graph statistics
    print(f"Graph Statistics:")
    print(f"- Number of triples: {len(g)}")
    print(f"- Number of nodes: {len(nx_graph.nodes())}")
    print(f"- Number of edges: {len(nx_graph.edges())}")
    print(f"- Number of unique subjects: {len(set(g.subjects()))}")
    print(f"- Number of unique predicates: {len(set(g.predicates()))}")
    print(f"- Number of unique objects: {len(set(g.objects()))}")
    
    return g, nx_graph