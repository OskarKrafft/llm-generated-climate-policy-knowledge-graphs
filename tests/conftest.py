import pytest
import os
import tempfile
import json
import shutil
import sys
from rdflib import Graph

# Add the project root to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)  # Make src importable

@pytest.fixture
def sample_article_dir():
    """Create a temporary directory with a comprehensive sample article for testing"""
    temp_dir = tempfile.mkdtemp()
    
    # Create policy_info.json
    policy_info = {
        "Titel": "TEST_Article_01",
        "CELEX_Number": "32000L0000"
    }
    with open(os.path.join(temp_dir, "policy_info.json"), "w", encoding="utf-8") as f:
        json.dump(policy_info, f)
    
    # Create Raw_Text.txt with comprehensive test content
    article_text = """
    This is a comprehensive test article about energy efficiency measures, renewable energy, and climate action.
    
    Member States shall establish national plans by 1 January 2025 to implement this directive.
    The Commission shall monitor implementation and report on progress by 31 December 2026.
    The European Parliament and the Council shall be informed of any significant delays.
    
    The Directive shall enter into force on the twentieth day following that of its publication.
    
    National reporting requirements shall be fulfilled every two years, with the first report 
    due on 15 May 2025.
    
    To achieve energy savings of at least 32.5% by 2030, Member States shall implement 
    mandatory energy labeling schemes, financial incentives, and public awareness campaigns.
    
    The Commission shall establish efficiency standards and conduct periodic reviews.
    
    Penalties for non-compliance shall be effective, proportionate and dissuasive.
    Member States shall notify the measures taken to ensure compliance to the Commission.
    
    Member States shall ensure that Eurostat receives all necessary data for statistical analysis.
    """
    
    with open(os.path.join(temp_dir, "Raw_Text.txt"), "w", encoding="utf-8") as f:
        f.write(article_text)
    
    # Create Curated_Annotations.jsonl with comprehensive annotations
    annotations = [
        # Instrument types
        {"span_id": "TEST1", "layer": "Instrumenttypes", "feature": "InstrumentType", "tag": "RegulatoryInstr", 
         "start": 23, "stop": 48, "text": "energy efficiency measures", "tokens": ["T1", "T2", "T3"]},
        {"span_id": "TEST2", "layer": "Instrumenttypes", "feature": "InstrumentType", "tag": "Edu_Outreach", 
         "start": 431, "stop": 457, "text": "public awareness campaigns", "tokens": ["T40", "T41", "T42"]},
        {"span_id": "TEST3", "layer": "Instrumenttypes", "feature": "InstrumentType", "tag": "TradablePermit", 
         "start": 343, "stop": 369, "text": "mandatory energy labeling", "tokens": ["T30", "T31", "T32"]},
        {"span_id": "TEST4", "layer": "Instrumenttypes", "feature": "InstrumentType", "tag": "Subsidies_Incentives", 
         "start": 378, "stop": 397, "text": "financial incentives", "tokens": ["T34", "T35"]},
        
        # Actors
        {"span_id": "TEST5", "layer": "Policydesigncharacteristics", "feature": "Actor", "tag": "Addressee_default", 
         "start": 112, "stop": 125, "text": "Member States", "tokens": ["T10", "T11"]},
        {"span_id": "TEST6", "layer": "Policydesigncharacteristics", "feature": "Actor", "tag": "Authority_default", 
         "start": 180, "stop": 190, "text": "Commission", "tokens": ["T17"]},
        {"span_id": "TEST7", "layer": "Policydesigncharacteristics", "feature": "Actor", "tag": "Authority_monitoring", 
         "start": 180, "stop": 190, "text": "Commission", "tokens": ["T17"]},
        {"span_id": "TEST8", "layer": "Policydesigncharacteristics", "feature": "Actor", "tag": "Authority_legislative", 
         "start": 239, "stop": 257, "text": "European Parliament", "tokens": ["T22", "T23"]},
        {"span_id": "TEST9", "layer": "Policydesigncharacteristics", "feature": "Actor", "tag": "Authority_default", 
         "start": 263, "stop": 270, "text": "Council", "tokens": ["T25"]},
        {"span_id": "TEST10", "layer": "Policydesigncharacteristics", "feature": "Actor", "tag": "Addressee_monitored", 
         "start": 112, "stop": 125, "text": "member\\xa0state", "tokens": ["T10", "T11"]},
        {"span_id": "TEST11", "layer": "Policydesigncharacteristics", "feature": "Actor", "tag": "Addressee_default", 
         "start": 597, "stop": 615, "text": "member\\xa0states", "tokens": ["T67", "T68", "T69"]},
        {"span_id": "TEST12", "layer": "Policydesigncharacteristics", "feature": "Actor", "tag": "Authority_established", 
         "start": 463, "stop": 473, "text": "Commission", "tokens": ["T45"]},
        {"span_id": "TEST13", "layer": "Policydesigncharacteristics", "feature": "Actor", "tag": "Addressee_default", 
         "start": 700, "stop": 713, "text": "Member States", "tokens": ["T77", "T78"]},
        {"span_id": "TEST14", "layer": "Policydesigncharacteristics", "feature": "Actor", "tag": "Addressee_resource", 
         "start": 725, "stop": 733, "text": "Eurostat", "tokens": ["T80"]},
        
        # Objectives
        {"span_id": "TEST15", "layer": "Policydesigncharacteristics", "feature": "Objective", "tag": "Objective_QualIntention", 
         "start": 31, "stop": 95, "text": "energy efficiency measures, renewable energy, and climate action", "tokens": ["T3", "T4", "T5", "T6", "T7", "T8", "T9"]},
        {"span_id": "TEST16", "layer": "Policydesigncharacteristics", "feature": "Objective", "tag": "Objective_QuantTarget", 
         "start": 313, "stop": 343, "text": "energy savings of at least 32.5% by 2030", "tokens": ["T27", "T28", "T29", "T30"]},
        
        # Time specifications
        {"span_id": "TEST17", "layer": "Policydesigncharacteristics", "feature": "Time", "tag": "Time_Compliance", 
         "start": 162, "stop": 175, "text": "1 January 2025", "tokens": ["T15", "T16"]},
        {"span_id": "TEST18", "layer": "Policydesigncharacteristics", "feature": "Time", "tag": "Time_Monitoring", 
         "start": 228, "stop": 243, "text": "31 December 2026", "tokens": ["T20", "T21"]},
        {"span_id": "TEST19", "layer": "Policydesigncharacteristics", "feature": "Time", "tag": "Time_InEffect", 
         "start": 302, "stop": 344, "text": "on the twentieth day following that of its publication", "tokens": ["T25", "T26", "T27", "T28", "T29"]},
        {"span_id": "TEST20", "layer": "Policydesigncharacteristics", "feature": "Time", "tag": "Time_Monitoring", 
         "start": 385, "stop": 401, "text": "every two years", "tokens": ["T35", "T36", "T37"]},
        {"span_id": "TEST21", "layer": "Policydesigncharacteristics", "feature": "Time", "tag": "Time_Compliance", 
         "start": 432, "stop": 444, "text": "15 May 2025", "tokens": ["T40", "T41", "T42"]},
        
        # Compliance - Monitoring
        {"span_id": "TEST22", "layer": "Policydesigncharacteristics", "feature": "Compliance", "tag": "Form_monitoring", 
         "start": 191, "stop": 224, "text": "monitor implementation and report on progress", "tokens": ["T18", "T19", "T20", "T21", "T22"]},
        {"span_id": "TEST23", "layer": "Policydesigncharacteristics", "feature": "Compliance", "tag": "Form_monitoring", 
         "start": 486, "stop": 505, "text": "conduct periodic reviews", "tokens": ["T47", "T48", "T49"]},
        {"span_id": "TEST24", "layer": "Policydesigncharacteristics", "feature": "Compliance", "tag": "Form_monitoring", 
         "start": 649, "stop": 694, "text": "notify the measures taken to ensure compliance", "tokens": ["T72", "T73", "T74", "T75", "T76", "T77"]},
        
        # Compliance - Sanctioning
        {"span_id": "TEST25", "layer": "Policydesigncharacteristics", "feature": "Compliance", "tag": "Form_sanctioning", 
         "start": 518, "stop": 567, "text": "Penalties for non-compliance shall be effective, proportionate", "tokens": ["T52", "T53", "T54", "T55", "T56", "T57", "T58"]},
        
        # ======== ANNOTATIONS THAT SHOULD BE IGNORED ========
        # Technology and application specificity features
        {"span_id": "IGN1", "layer": "Technologyandapplicationspecificity", "feature": "TechnologySpecificity", 
         "tag": "Tech_LowCarbon", "start": 50, "stop": 75, "text": "renewable energy sources", "tokens": ["T5", "T6", "T7"]},
        {"span_id": "IGN2", "layer": "Technologyandapplicationspecificity", "feature": "EnergySpecificity", 
         "tag": "Energy_LowCarbon", "start": 50, "stop": 66, "text": "renewable energy", "tokens": ["T5", "T6"]},
        {"span_id": "IGN3", "layer": "Technologyandapplicationspecificity", "feature": "ApplicationSpecificity", 
         "tag": "App_LowCarbon", "start": 76, "stop": 90, "text": "climate action", "tokens": ["T8", "T9"]},
        
        # Unspecified tags (should be ignored)
        {"span_id": "IGN4", "layer": "Instrumenttypes", "feature": "InstrumentType", 
         "tag": "Unspecified", "start": 518, "stop": 527, "text": "Penalties", "tokens": ["T52"]},
        
        # Time tags that should be ignored
        {"span_id": "IGN5", "layer": "Policydesigncharacteristics", "feature": "Time", 
         "tag": "Time_PolDuration", "start": 366, "stop": 381, "text": "every two years", "tokens": ["T35", "T36", "T37"]},
        {"span_id": "IGN6", "layer": "Policydesigncharacteristics", "feature": "Time", 
         "tag": "Time_Resources", "start": 386, "stop": 406, "text": "with the first report", "tokens": ["T38", "T39", "T40"]},
        
        # Addressee_sector with expected misspelling "Adressee_sector"
        {"span_id": "IGN7", "layer": "Policydesigncharacteristics", "feature": "Actor", 
         "tag": "Adressee_sector", "start": 313, "stop": 325, "text": "energy sector", "tokens": ["T27", "T28"]},
        
        # Reference and Resource features (should be ignored)
        {"span_id": "IGN8", "layer": "Policydesigncharacteristics", "feature": "Reference", 
         "tag": "Ref_OtherPolicy", "start": 176, "stop": 194, "text": "implement this directive", "tokens": ["T16", "T17", "T18"]},
        {"span_id": "IGN9", "layer": "Policydesigncharacteristics", "feature": "Resource", 
         "tag": "Resource_Information", "start": 725, "stop": 757, "text": "all necessary data for statistical analysis", "tokens": ["T80", "T81", "T82", "T83", "T84"]}
    ]
    
    with open(os.path.join(temp_dir, "Curated_Annotations.jsonl"), "w", encoding="utf-8") as f:
        for ann in annotations:
            f.write(json.dumps(ann) + "\n")
    
    yield temp_dir
    
    # Clean up after the test
    shutil.rmtree(temp_dir)

@pytest.fixture
def real_sample_article():
    """Use a real article from the test data directory"""
    # Point to an existing test article
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(project_root, "test_data", "EU_32006L0032_Title_0_Chapter_1_Section_0_Article_03")