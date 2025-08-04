#!/usr/bin/env python3
"""
CrewAI Database Schema Generator - MODULE-BY-MODULE APPROACH
=========================================================
"""

import warnings
import os
import time
import json
import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore", 
                       message=".*A custom validator is returning a value other than.*",
                       category=UserWarning)
warnings.filterwarnings("ignore", 
                       message=".*Returning anything other than.*",
                       category=UserWarning)
os.environ["PYDANTIC_DISABLE_WARNINGS"] = "1"

# CrewAI imports
try:
    from crewai import Agent, Task, Crew, Process
    from crewai_tools import FileReadTool
    from langchain_openai import ChatOpenAI
except ImportError as e:
    print("Missing dependencies. Please install:")
    print("pip install crewai==0.28.8 crewai-tools==0.1.6 langchain-openai==0.1.7 openai==1.12.0")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class"""
    # API Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    # Model Configuration
    primary_model: str = "gpt-4o-mini"
    advanced_model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 6000
    advanced_max_tokens: int = 8000
    
    # TPM Limits
    requests_per_minute: int = 50
    tokens_per_minute: int = 40000
    
    # Module-specific settings
    target_coverage_per_module: float = 0.95  # 95% coverage per module
    max_retries_per_module: int = 2
    
    # File paths
    transcript_file: str = "requirement_transcript.txt"
    visual_fields_file: str = "requirement_visual_fields.txt"
    
    # Output settings
    output_dir: str = "output"
    verbose: bool = True

class TPMManager:
    """Manages OpenAI API rate limits"""
    
    def __init__(self, config: Config):
        self.config = config
        self.request_count = 0
        self.token_count = 0
        self.last_reset = time.time()
        
    def check_and_wait(self, estimated_tokens: int = 2000):
        """Check TPM limits and wait if necessary"""
        current_time = time.time()
        
        # Reset counters every minute
        if current_time - self.last_reset >= 60:
            self.request_count = 0
            self.token_count = 0
            self.last_reset = current_time
            logger.info(" TPM counters reset")
            
        # Check if we need to wait
        if (self.request_count >= self.config.requests_per_minute or 
            self.token_count + estimated_tokens >= self.config.tokens_per_minute):
            wait_time = 60 - (current_time - self.last_reset)
            if wait_time > 0:
                logger.info(f" TPM limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                self.request_count = 0
                self.token_count = 0
                self.last_reset = time.time()
        
        self.request_count += 1
        self.token_count += estimated_tokens

class ModuleByModuleGenerator:
    """Main generator using proven module-by-module approach"""
    
    # MINIMAL HARDCODING: Only the known module structure from the document
    MODULE_DEFINITIONS = [
        {"page": "3", "name": "Module name", "expected_fields": 30},
        {"page": "4", "name": "Module description", "expected_fields": 20},
        {"page": "5", "name": "Module details", "expected_fields": 25},
        {"page": "6", "name": "Module relationships", "expected_fields": 15},
        {"page": "7", "name": "Module constraints", "expected_fields": 10},
        {"page": "8", "name": "Module fields overview", "expected_fields": 50},
        {"page": "9", "name": "Module finalization", "expected_fields": 46}
    ]
    
    def __init__(self, config: Config):
        self.config = config
        self.tpm_manager = TPMManager(config)
        
        # Validate configuration
        if not config.openai_api_key:
            raise ValueError(" OPENAI_API_KEY not found. Please set it as environment variable.")
        
        # Check source files
        if not os.path.exists(config.transcript_file):
            raise FileNotFoundError(f" {config.transcript_file} not found")
        if not os.path.exists(config.visual_fields_file):
            raise FileNotFoundError(f" {config.visual_fields_file} not found")
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize LLMs
        self.primary_llm = ChatOpenAI(
            model=config.primary_model,
            openai_api_key=config.openai_api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            request_timeout=60
        )
        
        self.advanced_llm = ChatOpenAI(
            model=config.advanced_model,
            openai_api_key=config.openai_api_key,
            temperature=config.temperature,
            max_tokens=config.advanced_max_tokens,
            request_timeout=90
        )
        
        # File reading tool
        self.file_tool = FileReadTool()
        
        # Results storage
        self.results = {
            "modules": {},
            "combined_schema": None,
            "final_sql": None
        }
    
    def create_module_extraction_agent(self) -> Agent:
        """Creates agent specialized in extracting single modules completely"""
        return Agent(
            role="Module Field Extraction Specialist",
            goal="Extract EVERY field from a single module with 95%+ completeness",
            backstory="""You are a specialist who focuses on one module at a time to achieve 
            perfect field extraction. You systematically read through each section of a module 
            and extract every field definition, data type, and constraint. You never rush or 
            skip sections - you are thorough and methodical. You achieve 95%+ field coverage 
            on every module because you focus completely on one area at a time.""",
            verbose=True,
            allow_delegation=False,
            llm=self.primary_llm,
            tools=[self.file_tool]
        )
    
    def create_module_validator_agent(self) -> Agent:
        """Creates agent specialized in validating single module extraction"""
        return Agent(
            role="Module Completeness Validator",
            goal="Validate that a module extraction captured ALL fields from the source",
            backstory="""You are a quality control specialist who validates field extractions 
            against the source document. You count every field mentioned in a module section 
            and compare against what was extracted. You identify specific missing fields by 
            name and location. You ensure no field is missed.""",
            verbose=True,
            allow_delegation=False,
            llm=self.primary_llm,
            tools=[self.file_tool]
        )
    
    def create_schema_integration_agent(self) -> Agent:
        """Creates agent specialized in combining modules into final schema"""
        return Agent(
            role="Schema Integration Architect", 
            goal="Combine all module extractions into a complete, consistent SQL Server schema",
            backstory="""You are a database architect who takes individual module extractions 
            and combines them into a unified schema. You ensure consistency across modules, 
            proper relationships, and generate complete SQL Server DDL. You never omit fields 
            or create partial schemas.""",
            verbose=True,
            allow_delegation=False,
            llm=self.advanced_llm
        )
    
    def extract_single_module(self, module_def: Dict[str, Any], retry_count: int = 0) -> Dict[str, Any]:
        """Extract fields from a single module with validation"""
        module_name = module_def["name"]
        page_num = module_def["page"]
        expected_fields = module_def["expected_fields"]
        
        logger.info(f" Extracting {module_name} module (Page {page_num})...")
        logger.info(f" Target: {expected_fields} fields")
        
        extraction_agent = self.create_module_extraction_agent()
        
        extraction_task = Task(
            description=f"""
            WORKFLOW:
            1. Read "{self.config.visual_fields_file}" once and keep in memory
            2. Read "{self.config.transcript_file}" once and keep in memory for context
            3. From memory, process only the PAGE {page_num} section from visual_fields
            4. Use transcript context from memory as needed
            5. Extract fields and output JSON immediately
            6. Do NOT read any files again after initial reads

            EXTRACTION REQUIREMENTS:
            1. From the loaded content, locate PAGE {page_num} section
            2. Extract EVERY field mentioned in ALL sections of this module:
               - List View fields
               - Detail Form fields  
               - Sub-sections and tabs
               - Boolean flags and category fields
               - Address sections (visiting vs postal)
               - Audit/control fields
            3. For each field capture:
               - Exact field name
               - Proper SQL Server data type
               - All constraints (PRIMARY KEY, NOT NULL, etc.)
               - Field description
               - Source location (PAGE {page_num})
            
            TARGET: Extract {expected_fields} fields from {module_name}
            
            FOCUS AREAS FOR {module_name}:
            - Do NOT extract from other pages
            - Do NOT skip subsections
            - Include ALL visible columns from list views
            - Include ALL form fields from detail views
            - Include complex structures and grids
            
            OUTPUT FORMAT:
            {{
                "module_name": "{module_name}",
                "page": "{page_num}",
                "fields": [
                    {{
                        "name": "exact_field_name",
                        "data_type": "SQL_SERVER_TYPE",
                        "constraints": ["constraint_list"],
                        "description": "field_description",
                        "source_location": "PAGE {page_num}: section_name"
                    }}
                ],
                "primary_key": ["key_fields"],
                "relationships": ["foreign_key_specs"],
                "total_fields_extracted": number
            }}
            """,
            expected_output=f"Complete JSON extraction of {module_name} module with all {expected_fields} fields",
            agent=extraction_agent
        )
        
        extraction_crew = Crew(
            agents=[extraction_agent],
            tasks=[extraction_task],
            process=Process.sequential,
            verbose=True
        )
        
        self.tpm_manager.check_and_wait(3000)
        
        try:
            result = extraction_crew.kickoff()
            
            # Safe result extraction
            result_text = self.safe_extract_text(result)
            
            # Validate extraction
            validation_result = self.validate_module_extraction(result_text, module_def)
            
            if validation_result["coverage"] >= self.config.target_coverage_per_module:
                logger.info(f" {module_name}: {validation_result['extracted_fields']}/{expected_fields} fields ({validation_result['coverage']:.1%})")
                return {
                    "module": module_def,
                    "extraction": result_text,
                    "validation": validation_result,
                    "success": True
                }
            elif retry_count < self.config.max_retries_per_module:
                logger.warning(f" {module_name}: Only {validation_result['coverage']:.1%} coverage. Retrying...")
                return self.extract_single_module(module_def, retry_count + 1)
            else:
                logger.warning(f" {module_name}: {validation_result['coverage']:.1%} coverage after {retry_count + 1} attempts")
                return {
                    "module": module_def,
                    "extraction": result_text,
                    "validation": validation_result,
                    "success": False
                }
                
        except Exception as e:
            logger.error(f" Error extracting {module_name}: {str(e)}")
            if retry_count < self.config.max_retries_per_module:
                logger.info(f"🔄 Retrying {module_name}...")
                return self.extract_single_module(module_def, retry_count + 1)
            else:
                return {
                    "module": module_def,
                    "extraction": f"Error: {str(e)}",
                    "validation": {"coverage": 0, "extracted_fields": 0},
                    "success": False
                }
    
    def validate_module_extraction(self, extraction_result: str, module_def: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a module extraction for completeness"""
        module_name = module_def["name"]
        expected_fields = module_def["expected_fields"]
        
        try:
            # Try to extract field count from JSON
            if "{" in extraction_result and "}" in extraction_result:
                # Look for field arrays in JSON
                field_count_match = re.search(r'"fields":\s*\[\s*(?:.*?\}.*?)*\]', extraction_result, re.DOTALL)
                if field_count_match:
                    # Count field objects in the array
                    field_objects = re.findall(r'\{\s*"name":', field_count_match.group())
                    extracted_count = len(field_objects)
                else:
                    # Fallback: look for "total_fields_extracted"
                    count_match = re.search(r'"total_fields_extracted":\s*(\d+)', extraction_result)
                    extracted_count = int(count_match.group(1)) if count_match else 0
            else:
                # Fallback: count field mentions
                field_mentions = re.findall(r'"name":\s*"[^"]+', extraction_result)
                extracted_count = len(field_mentions)
            
            coverage = extracted_count / expected_fields if expected_fields > 0 else 0
            
            return {
                "coverage": coverage,
                "extracted_fields": extracted_count,
                "expected_fields": expected_fields,
                "module": module_name
            }
            
        except Exception as e:
            logger.warning(f"Validation error for {module_name}: {str(e)}")
            return {
                "coverage": 0,
                "extracted_fields": 0,
                "expected_fields": expected_fields,
                "module": module_name,
                "error": str(e)
            }
    
    def combine_modules_into_schema(self, module_results: List[Dict[str, Any]]) -> str:
        """Combine all module extractions into final schema"""
        logger.info(" Combining all modules into unified schema...")
        
        integration_agent = self.create_schema_integration_agent()
        
        # Prepare all module extractions for combination
        combined_extractions = {}
        total_fields = 0
        
        for result in module_results:
            if result["success"]:
                module_name = result["module"]["name"]
                combined_extractions[module_name] = result["extraction"]
                total_fields += result["validation"]["extracted_fields"]
        
        integration_task = Task(
            description=f"""
            COMBINE ALL MODULE EXTRACTIONS INTO COMPLETE SQL SERVER SCHEMA:
            
            MODULE EXTRACTIONS TO COMBINE:
            {json.dumps(combined_extractions, indent=2)}
            
            TOTAL FIELDS TO INCLUDE: {total_fields}
            
            INTEGRATION REQUIREMENTS:
            1. Create complete SQL Server DDL for ALL modules
            2. Include EVERY field from EVERY module extraction
            3. Use proper SQL Server data types:
               - VARCHAR for strings
               - BIT for booleans  
               - INT for integers
               - DECIMAL(10,2) for prices/amounts
               - DATE for dates
               - TEXT for large text fields
            4. Add all PRIMARY KEY constraints
            5. Add FOREIGN KEY relationships between tables
            6. Include descriptive comments for each field
            7. Generate complete, production-ready DDL
            
            ENTITY RELATIONSHIPS:
            - Ensure all foreign keys are properly defined
            - Maintain referential integrity across modules
            
            OUTPUT: Complete SQL Server DDL script with all entities and all fields
            """,
            expected_output="Complete SQL Server DDL script with all extracted fields",
            agent=integration_agent
        )
        
        integration_crew = Crew(
            agents=[integration_agent],
            tasks=[integration_task],
            process=Process.sequential,
            verbose=True
        )
        
        self.tpm_manager.check_and_wait(4000)
        result = integration_crew.kickoff()
        
        return self.safe_extract_text(result)
    
    def safe_extract_text(self, result: Any) -> str:
        """Safely extract text from CrewAI result object"""
        try:
            if hasattr(result, 'raw'):
                return str(result.raw)
            elif hasattr(result, 'output'):
                return str(result.output)
            elif hasattr(result, 'content'):
                return str(result.content)
            else:
                return str(result)
        except Exception as e:
            logger.warning(f"Failed to extract text from result: {e}")
            return str(result)
    
    def save_results(self):
        """Save all results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save module results
            module_file = f"{self.config.output_dir}/module_extractions_{timestamp}.json"
            with open(module_file, "w", encoding='utf-8') as f:
                json.dump(self.results["modules"], f, indent=2, default=str)
            logger.info(f" Module extractions saved: {module_file}")
            
            # Save final schema
            if self.results["final_sql"]:
                schema_file = f"{self.config.output_dir}/complete_schema_{timestamp}.sql"
                with open(schema_file, "w", encoding='utf-8') as f:
                    f.write(str(self.results["final_sql"]))
                logger.info(f" Final schema saved: {schema_file}")
            
            # Save summary
            summary_file = f"{self.config.output_dir}/generation_summary_{timestamp}.json"
            summary = {
                "timestamp": timestamp,
                "approach": "module_by_module",
                "modules_processed": len(self.results["modules"]),
                "total_fields_extracted": sum(m.get("validation", {}).get("extracted_fields", 0) 
                                            for m in self.results["modules"].values()),
                "overall_success": all(m.get("success", False) for m in self.results["modules"].values()),
                "config": {
                    "primary_model": self.config.primary_model,
                    "advanced_model": self.config.advanced_model,
                    "target_coverage": self.config.target_coverage_per_module
                }
            }
            
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f" Summary saved: {summary_file}")
            
        except Exception as e:
            logger.error(f" Error saving results: {str(e)}")
    
    def run_module_by_module_process(self) -> Dict[str, Any]:
        """Execute the proven module-by-module extraction process"""
        start_time = time.time()
        
        try:
            logger.info(" Starting MODULE-BY-MODULE Database Schema Generation")
            logger.info(f" Target: {len(self.MODULE_DEFINITIONS)} modules")
            logger.info(f" Expected total fields: {sum(m['expected_fields'] for m in self.MODULE_DEFINITIONS)}")
            
            # Phase 1: Extract each module individually
            logger.info(" Phase 1: Module-by-module extraction...")
            module_results = []
            
            for module_def in self.MODULE_DEFINITIONS:
                result = self.extract_single_module(module_def)
                module_results.append(result)
                self.results["modules"][module_def["name"]] = result
                
                # Brief pause between modules
                time.sleep(1)
            
            # Phase 2: Combine into unified schema
            logger.info(" Phase 2: Schema integration...")
            final_schema = self.combine_modules_into_schema(module_results)
            self.results["final_sql"] = final_schema
            
            # Calculate final statistics
            total_extracted = sum(r["validation"]["extracted_fields"] for r in module_results)
            total_expected = sum(m["expected_fields"] for m in self.MODULE_DEFINITIONS)
            overall_coverage = total_extracted / total_expected if total_expected > 0 else 0
            successful_modules = sum(1 for r in module_results if r["success"])
            
            # Save all results
            self.save_results()
            
            processing_time = time.time() - start_time
            
            logger.info(f" MODULE-BY-MODULE process completed in {processing_time:.1f} seconds!")
            logger.info(f" Results: {total_extracted}/{total_expected} fields ({overall_coverage:.1%} coverage)")
            logger.info(f" Successful modules: {successful_modules}/{len(self.MODULE_DEFINITIONS)}")
            
            return {
                'success': overall_coverage >= 0.85,  # 85% minimum for success
                'processing_time': processing_time,
                'total_fields_extracted': total_extracted,
                'total_fields_expected': total_expected,
                'overall_coverage': overall_coverage,
                'successful_modules': successful_modules,
                'total_modules': len(self.MODULE_DEFINITIONS),
                'results': self.results,
                'output_dir': self.config.output_dir
            }
            
        except Exception as e:
            logger.error(f" MODULE-BY-MODULE process failed: {str(e)}")
            
            try:
                self.save_results()
                logger.info(" Partial results saved")
            except:
                pass
            
            return {
                'success': False,
                'error': str(e),
                'partial_results': self.results
            }

def print_banner():
    """Print application banner"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║    CrewAI Schema Generator - MODULE-BY-MODULE APPROACH      ║
╚══════════════════════════════════════════════════════════════╝
""")

def check_prerequisites(config: Config) -> List[str]:
    """Check if all prerequisites are met"""
    issues = []
    
    if not config.openai_api_key:
        issues.append(" OPENAI_API_KEY environment variable not set")
    
    if not os.path.exists(config.transcript_file):
        issues.append(f" {config.transcript_file} not found")
    
    if not os.path.exists(config.visual_fields_file):
        issues.append(f" {config.visual_fields_file} not found")
    
    return issues

def main():
    """Main execution function"""
    print_banner()
    
    # Load configuration
    config = Config()
    
    # Check prerequisites
    issues = check_prerequisites(config)
    if issues:
        print(" Prerequisites check failed:")
        for issue in issues:
            print(f"   {issue}")
        print("\nPlease resolve these issues and try again.")
        return False
    
    print(" All prerequisites met!")
    
    # Ask for confirmation
    try:
        response = input("\n Start module-by-module generation? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Operation cancelled.")
            return False
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return False
    
    # Initialize and run generator
    try:
        generator = ModuleByModuleGenerator(config)
        result = generator.run_module_by_module_process()
        
        if result['success']:
            print(f"\n SUCCESS! Process completed in {result['processing_time']:.1f} seconds")
            print(f" Final coverage: {result['overall_coverage']:.1%} ({result['total_fields_extracted']}/{result['total_fields_expected']} fields)")
            print(f" Successful modules: {result['successful_modules']}/{result['total_modules']}")
            print(f" Results saved in: {result['output_dir']}/")
            print("\n Generated files:")
            print("   • module_extractions_*.json (detailed module results)")
            print("   • complete_schema_*.sql (COMPLETE SQL Server DDL)")
            print("   • generation_summary_*.json (process summary)")
            return True
        else:
            print(f"\n FAILED: {result.get('error', 'Unknown error')}")
            if 'partial_results' in result:
                print(" Partial results may have been saved")
            return False
            
    except KeyboardInterrupt:
        print("\n  Process interrupted by user")
        return False
    except Exception as e:
        print(f"\n Unexpected error: {str(e)}")
        logger.exception("Full error details:")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)