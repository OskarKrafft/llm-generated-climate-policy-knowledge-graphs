import os
import json
import re

def load_ontology(ontology_path):
    """Load the ontology file and return its contents"""
    try:
        with open(ontology_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Ontology file not found at {ontology_path}")
        return None

def create_system_prompt_with_ontology(system_template, ontology_path):
    """Insert the ontology content into the system template at {ontology} placeholder"""
    ontology_content = load_ontology(ontology_path)
    if not ontology_content:
        print("Warning: Using system prompt without ontology content")
        return system_template
    
    escaped_ontology_content = ontology_content.replace("{", "{{").replace("}", "}}")
        
    # Replace the placeholder with the actual ontology content
    return system_template.replace("{ontology}", escaped_ontology_content)

def load_prompt_template(template_name, prompt_dir=None):
    """Load a prompt template from the specified prompts directory"""
    if not prompt_dir:
        # Fallback to default if no directory is provided
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        prompt_dir = os.path.join(project_root, "experiment-1/prompts")
        print(f"Warning: No prompt directory specified, defaulting to {prompt_dir}")
        
    template_path = os.path.join(prompt_dir, f"{template_name}.txt")
    
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # Don't print error here, let the caller handle fallback logic
        return None

def format_prompt(template, **kwargs):
    """Format a prompt template with the given parameters"""
    if template is None:
        return None
    
    try:
        return template.format(**kwargs)
    except KeyError as e:
        print(f"Missing parameter in prompt template: {e}")
        return None

def create_conversation(system_template=None, user_template=None, examples=None, **kwargs):
    """
    Create a conversation array for LLM clients using templates and examples.
    
    Args:
        system_template: Template string for system message
        user_template: Template string for user message
        examples: List of dicts with 'user' and 'assistant' keys for few-shot examples
        **kwargs: Parameters to format the templates
        
    Returns:
        List of message dicts ready for LLM client
    """
    conversation = []
    
    # Add system message if provided
    if system_template:
        system_content = format_prompt(system_template, **kwargs)
        if system_content:
            conversation.append({"role": "system", "content": system_content})
    
    # Add few-shot examples if provided
    if examples:
        for example in examples:
            if 'user' in example:
                # Use example['user'] instead of user_content
                conversation.append({"role": "user", "content": example['user']})
            if 'assistant' in example:
                conversation.append({"role": "assistant", "content": example['assistant']})
    
    # Add final user message
    if user_template:
        user_content = format_prompt(user_template, **kwargs)
        if user_content:
            conversation.append({"role": "user", "content": user_content})
    
    return conversation

def get_prompt_strategy(strategy_name, ontology_path=None, prompt_dir=None, **kwargs):
    """
    Get a prepared conversation using a named prompting strategy with dynamic ontology.
    
    Args:
        strategy_name: 'zero-shot', 'one-shot', or 'few-shot'
        ontology_path: Path to the TTL file containing the ontology
        prompt_dir: Path to the directory containing prompt templates
        **kwargs: Parameters for templates (e.g., article_text, policy_info, output_format)
        
    Returns:
        List of message dicts ready for LLM client
    """
    output_format = kwargs.get('output_format', 'ttl').lower()
    # For raw JSON triplet output, use raw-specific prompt templates
    if output_format == 'raw':
        system_base = 'system_prompt_raw'
        user_base = 'user_prompt_raw'
        one_shot_base = 'example_one_shot_raw'
        few_shot_base = 'examples_few_shot_raw'
    else:
        # Normalize jsonld
        if output_format == 'jsonld':
            output_format = 'json-ld'
        # Determine base template names based on format
        system_base = 'system_prompt'
        user_base = 'user_prompt'
        one_shot_base = 'example_one_shot'
        few_shot_base = 'examples_few_shot'
        if output_format == 'json-ld':
            system_base = 'system_prompt_jsonld'
            user_base = 'user_prompt_jsonld'
            one_shot_base = 'example_one_shot_jsonld'
            few_shot_base = 'examples_few_shot_jsonld'

    # Load the base system prompt template
    system_template = load_prompt_template(system_base, prompt_dir)
    # Fallback to default if format-specific template not found
    if not system_template and output_format == 'json-ld':
        print(f"Warning: {system_base}.txt not found in {prompt_dir}, falling back to system_prompt.txt")
        system_template = load_prompt_template("system_prompt", prompt_dir)

    # Load and insert the ontology if a path is provided
    if ontology_path and system_template:
        system_template = create_system_prompt_with_ontology(system_template, ontology_path)
    
    # Load the user prompt template
    user_template = load_prompt_template(user_base, prompt_dir)
    # Fallback to default if format-specific template not found
    if not user_template and output_format == 'json-ld':
        print(f"Warning: {user_base}.txt not found in {prompt_dir}, falling back to user_prompt.txt")
        user_template = load_prompt_template("user_prompt", prompt_dir)
    
    examples = []
    
    if strategy_name == 'one-shot':
        # Load the appropriate one-shot example
        example_data = load_prompt_template(one_shot_base, prompt_dir)
        # Fallback to default if format-specific example not found
        if not example_data and output_format == 'json-ld':
            print(f"Warning: {one_shot_base}.txt not found in {prompt_dir}, falling back to example_one_shot.txt")
            example_data = load_prompt_template("example_one_shot", prompt_dir)
            
        if example_data:
            try:
                example_dict = json.loads(example_data)
                examples = [example_dict]
            except json.JSONDecodeError:
                print(f"Error: {one_shot_base}.txt (or fallback) is not valid JSON")
    
    elif strategy_name == 'few-shot':
        # Load the appropriate few-shot examples
        examples_data = load_prompt_template(few_shot_base, prompt_dir)
        # Fallback to default if format-specific examples not found
        if not examples_data and output_format == 'json-ld':
            print(f"Warning: {few_shot_base}.txt not found in {prompt_dir}, falling back to examples_few_shot.txt")
            examples_data = load_prompt_template("examples_few_shot", prompt_dir)
            
        if examples_data:
            try:
                examples = json.loads(examples_data)
            except json.JSONDecodeError:
                print(f"Error: {few_shot_base}.txt (or fallback) is not valid JSON")
    
    return create_conversation(system_template, user_template, examples, **kwargs)