import streamlit as st
import logging
import json
import requests
import re
import os
import datetime
import pandas as pd
import altair as alt
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO,                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Merged Functions and UI from document_categorization (2).py and (3).py ---

def document_categorization():
    """
    Enhanced document categorization with improved confidence metrics and user-defined types.
    (Merged from versions 2 and 3)
    """
    st.title("Document Categorization")
    
    if not st.session_state.authenticated or not st.session_state.client:
        st.error("Please authenticate with Box first")
        return
    
    if not st.session_state.selected_files:
        st.warning("No files selected. Please select files in the File Browser first.")
        if st.button("Go to File Browser", key="go_to_file_browser_button_cat"):
            st.session_state.current_page = "File Browser"
            st.rerun()
        return
    
    # Initialize document categorization state if not exists (from version 3)
    if "document_categorization" not in st.session_state:
        st.session_state.document_categorization = {
            "is_categorized": False,
            "results": {},
            "errors": {}
        }
    
    # Initialize confidence thresholds if not exists (from version 3)
    if "confidence_thresholds" not in st.session_state:
        st.session_state.confidence_thresholds = {
            "auto_accept": 0.85,
            "verification": 0.6,
            "rejection": 0.4
        }
        
    # Initialize document types if not exists (NEW STRUCTURE from version 2)
    if "document_types" not in st.session_state or not isinstance(st.session_state.document_types, list) or not all(isinstance(item, dict) and "name" in item and "description" in item for item in st.session_state.document_types):
        logger.warning("Initializing/Resetting document_types in session state to default structure.")
        st.session_state.document_types = [
            {"name": "Sales Contract", "description": "Contracts related to sales agreements and terms."},
            {"name": "Invoices", "description": "Billing documents issued by a seller to a buyer, indicating quantities, prices for products or services."},
            {"name": "Tax", "description": "Documents related to government taxation (e.g., tax forms, filings, receipts)."},
            {"name": "Financial Report", "description": "Reports detailing the financial status or performance of an entity (e.g., balance sheets, income statements)."},
            {"name": "Employment Contract", "description": "Agreements outlining terms and conditions of employment between an employer and employee."},
            {"name": "PII", "description": "Documents containing Personally Identifiable Information that needs careful handling."},
            {"name": "Other", "description": "Any document not fitting into the specific categories above."}
        ]
    
    # Display selected files
    num_files = len(st.session_state.selected_files)
    st.write(f"Ready to categorize {num_files} files using Box AI.")
    
    # Create tabs for main interface and settings (from version 2)
    tab1, tab2 = st.tabs(["Categorization", "Settings"])
    
    with tab1:
        # AI Model selection (Preserved from version 3)
        # Updated list of models with descriptions - FILTERED FOR /ai/ask
        all_models_with_desc = {
            "azure__openai__gpt_4_1_mini": "Azure OpenAI GPT-4.1 Mini: Lightweight multimodal model (Default for Box AI for Docs/Notes Q&A)",
            "google__gemini_2_0_flash_lite_preview": "Google Gemini 2.0 Flash Lite: Lightweight multimodal model (Preview)",
            "azure__openai__gpt_4o_mini": "Azure OpenAI GPT-4o Mini: Lightweight multimodal model",
            "azure__openai__gpt_4o": "Azure OpenAI GPT-4o: Highly efficient multimodal model for complex tasks", # Not in allowedValues
            "azure__openai__gpt_4_1": "Azure OpenAI GPT-4.1: Highly efficient multimodal model for complex tasks",
            "azure__openai__gpt_o3": "Azure OpenAI GPT o3: Highly efficient multimodal model for complex tasks", # Not in allowedValues
            "azure__openai__gpt_o4-mini": "Azure OpenAI GPT o4-mini: Highly efficient multimodal model for complex tasks", # Not in allowedValues
            "google__gemini_2_5_pro_preview": "Google Gemini 2.5 Pro: Optimal for high-volume, high-frequency tasks (Preview)", # Not in allowedValues
            "google__gemini_2_5_flash_preview": "Google Gemini 2.5 Flash: Optimal for high-volume, high-frequency tasks (Preview)", # Not in allowedValues
            "google__gemini_2_0_flash_001": "Google Gemini 2.0 Flash: Optimal for high-volume, high-frequency tasks",
            "google__gemini_1_5_flash_001": "Google Gemini 1.5 Flash: High volume tasks & latency-sensitive applications",
            "google__gemini_1_5_pro_001": "Google Gemini 1.5 Pro: Foundation model for various multimodal tasks",
            "aws__claude_3_haiku": "AWS Claude 3 Haiku: Tailored for various language tasks",
            "aws__claude_3_sonnet": "AWS Claude 3 Sonnet: Advanced language tasks, comprehension & context handling",
            "aws__claude_3_5_sonnet": "AWS Claude 3.5 Sonnet: Enhanced language understanding and generation",
            "aws__claude_3_7_sonnet": "AWS Claude 3.7 Sonnet: Enhanced language understanding and generation",
            "aws__titan_text_lite": "AWS Titan Text Lite: Advanced language processing, extensive contexts",
            "ibm__llama_3_2_instruct": "IBM Llama 3.2 Instruct: Instruction-tuned text model for dialogue, retrieval, summarization", # Renamed in error log?
            "ibm__llama_3_2_90b_vision_instruct": "IBM Llama 3.2 90B Vision Instruct: Instruction-tuned vision model (From Error Log)", # Added from error log
            "ibm__llama_4_scout": "IBM Llama 4 Scout: Natively multimodal model for text and multimodal experiences",
            "xai__grok_3_beta": "xAI Grok 3: Excels at data extraction, coding, summarization (Beta)", # Not in allowedValues
            "xai__grok_3_mini_beta": "xAI Grok 3 Mini: Lightweight model for logic-based tasks (Beta)" # Not in allowedValues
        }
        allowed_model_names = [
            "azure__openai__gpt_4o_mini", "azure__openai__gpt_4_1", "azure__openai__gpt_4_1_mini",
            "google__gemini_1_5_pro_001", "google__gemini_1_5_flash_001", "google__gemini_2_0_flash_001",
            "google__gemini_2_0_flash_lite_preview", "aws__claude_3_haiku", "aws__claude_3_sonnet",
            "aws__claude_3_5_sonnet", "aws__claude_3_7_sonnet", "aws__titan_text_lite",
            "ibm__llama_3_2_90b_vision_instruct", "ibm__llama_4_scout"
        ]
        ai_models_with_desc = {name: all_models_with_desc.get(name, f"{name} (Description not found)")
                               for name in allowed_model_names if name in all_models_with_desc}
        for name in allowed_model_names:
            if name not in ai_models_with_desc:
                 ai_models_with_desc[name] = f"{name} (Description not found)"
                 logger.warning(f"Model \t{name}\t from allowed list was missing description, added placeholder.")
        ai_model_names = list(ai_models_with_desc.keys())
        ai_model_options = list(ai_models_with_desc.values())
        if "categorization_ai_model" not in st.session_state:
            st.session_state.categorization_ai_model = ai_model_names[0]
        current_model_name = st.session_state.categorization_ai_model
        if current_model_name not in ai_model_names:
            logger.warning(f"Previously selected categorization model \t{current_model_name}\t is not allowed. Defaulting to \t{ai_model_names[0]}\t.")
            current_model_name = ai_model_names[0]
            st.session_state.categorization_ai_model = current_model_name
        try:
            current_model_desc = ai_models_with_desc.get(current_model_name, ai_model_options[0])
            selected_index = ai_model_options.index(current_model_desc)
        except (ValueError, KeyError):
            logger.error(f"Error finding index for categorization model \t{current_model_name}\t. Defaulting to first model.")
            selected_index = 0
            current_model_name = ai_model_names[selected_index]
            st.session_state.categorization_ai_model = current_model_name
        selected_model_desc = st.selectbox(
            "Select AI Model for Categorization",
            options=ai_model_options,
            index=selected_index,
            key="ai_model_select_cat",
            help="Choose the AI model for categorization. Only models supported by the Q&A endpoint are listed."
        )
        selected_model_name = ""
        for name, desc in ai_models_with_desc.items():
            if desc == selected_model_desc:
                selected_model_name = name
                break
        st.session_state.categorization_ai_model = selected_model_name
        selected_model = selected_model_name
        
        # Enhanced categorization options (Preserved from version 3)
        st.write("### Categorization Options")
        col1_opt, col2_opt = st.columns(2)
        with col1_opt:
            use_two_stage = st.checkbox(
                "Use two-stage categorization",
                value=True,
                key="use_two_stage_cat",
                help="When enabled, documents with low confidence will undergo a second analysis"
            )
            use_consensus = st.checkbox(
                "Use multi-model consensus",
                value=False,
                key="use_consensus_cat",
                help="When enabled, multiple AI models will be used and their results combined for more accurate categorization"
            )
        with col2_opt:
            confidence_threshold = st.slider(
                "Confidence threshold for second-stage",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
                key="confidence_threshold_cat",
                help="Documents with confidence below this threshold will undergo second-stage analysis",
                disabled=not use_two_stage
            )
            consensus_models = []
            if use_consensus:
                selected_consensus_descs = st.multiselect(
                    "Select models for consensus",
                    options=ai_model_options,
                    default=[ai_model_options[0], ai_model_options[1]] if len(ai_model_options) > 1 else ai_model_options[:1],
                    help="Select 2-3 models for best results (more models will increase processing time)",
                    key="consensus_models_multiselect"
                )
                consensus_models = []
                for desc in selected_consensus_descs:
                    for name, description in ai_models_with_desc.items():
                        if description == desc:
                            consensus_models.append(name)
                            break
                if len(consensus_models) < 1:
                    st.warning("Please select at least one model for consensus categorization")
        
        # Categorization controls
        col1_ctrl, col2_ctrl = st.columns(2)
        with col1_ctrl:
            start_button = st.button("Start Categorization", key="start_categorization_button_cat", use_container_width=True)
        with col2_ctrl:
            cancel_button = st.button("Cancel Categorization", key="cancel_categorization_button_cat", use_container_width=True)
        
        # Process categorization
        if start_button:
            # Get the current list of document type names from session state
            current_doc_types = st.session_state.get("document_types", [])
            valid_categories = [dtype["name"] for dtype in current_doc_types if isinstance(dtype, dict) and "name" in dtype]
            
            if not valid_categories:
                 st.error("Cannot start categorization: No valid document types defined in Settings.")
            else:
                with st.spinner("Categorizing documents..."):
                    # Reset categorization results (from version 3)
                    st.session_state.document_categorization = {
                        "is_categorized": False,
                        "results": {},
                        "errors": {}
                    }
                    
                    # Process each file (Logic merged from v2 and v3)
                    for file in st.session_state.selected_files:
                        file_id = file["id"]
                        file_name = file["name"]
                        
                        try:
                            if use_consensus and consensus_models:
                                consensus_results = []
                                model_progress = st.progress(0)
                                model_status = st.empty()
                                for i, model in enumerate(consensus_models):
                                    model_status.text(f"Processing with {model}...")
                                    # Pass dynamic category names
                                    result = categorize_document(file_id, model, valid_categories)
                                    consensus_results.append(result)
                                    model_progress.progress((i + 1) / len(consensus_models))
                                model_progress.empty()
                                model_status.empty()
                                result = combine_categorization_results(consensus_results)
                                models_text = ", ".join(consensus_models)
                                result["reasoning"] = f"Consensus from models: {models_text}\n\n" + result["reasoning"]
                            else:
                                # Pass dynamic category names
                                result = categorize_document(file_id, selected_model, valid_categories)
                                if use_two_stage and result["confidence"] < confidence_threshold:
                                    st.info(f"Low confidence ({result["confidence"]:.2f}) for {file_name}, performing detailed analysis...")
                                    # Pass dynamic category names
                                    detailed_result = categorize_document_detailed(file_id, selected_model, result["document_type"], valid_categories)
                                    result = {
                                        "document_type": detailed_result["document_type"],
                                        "confidence": detailed_result["confidence"],
                                        "reasoning": detailed_result["reasoning"],
                                        "first_stage_type": result["document_type"],
                                        "first_stage_confidence": result["confidence"]
                                    }
                            
                            document_features = extract_document_features(file_id)
                            # Pass dynamic category names
                            multi_factor_confidence = calculate_multi_factor_confidence(
                                result["confidence"],
                                document_features,
                                result["document_type"],
                                result.get("reasoning", ""),
                                valid_categories
                            )
                            calibrated_confidence = apply_confidence_calibration(
                                result["document_type"],
                                multi_factor_confidence["overall"]
                            )
                            
                            # Store result structure from version 2
                            st.session_state.document_categorization["results"][file_id] = {
                                "file_id": file_id,
                                "file_name": file_name,
                                "document_type": result["document_type"],
                                "confidence": result["confidence"],  # Original AI confidence
                                "multi_factor_confidence": multi_factor_confidence,  # Detailed confidence factors
                                "calibrated_confidence": calibrated_confidence,  # Calibrated overall confidence
                                "reasoning": result["reasoning"],
                                "first_stage_type": result.get("first_stage_type"),
                                "first_stage_confidence": result.get("first_stage_confidence"),
                                "document_features": document_features
                            }
                        except Exception as e:
                            logger.error(f"Error categorizing document {file_name}: {str(e)}")
                            # Store error structure from version 2
                            st.session_state.document_categorization["errors"][file_id] = {
                                "file_id": file_id,
                                "file_name": file_name,
                                "error": str(e)
                            }
                    
                    # Apply confidence thresholds (Function from version 2)
                    st.session_state.document_categorization["results"] = apply_confidence_thresholds(
                        st.session_state.document_categorization["results"]
                    )
                    
                    st.session_state.document_categorization["is_categorized"] = True
                    num_processed = len(st.session_state.document_categorization["results"])
                    num_errors = len(st.session_state.document_categorization["errors"])
                    if num_errors == 0:
                        st.success(f"Categorization complete! Processed {num_processed} files.")
                    else:
                        st.warning(f"Categorization complete! Processed {num_processed} files with {num_errors} errors.")
                    st.rerun() # Rerun to update display
        
        # Display categorization results
        if st.session_state.document_categorization.get("is_categorized", False):
            # Pass dynamic category names
            current_doc_types = st.session_state.get("document_types", [])
            valid_categories = [dtype["name"] for dtype in current_doc_types if isinstance(dtype, dict) and "name" in dtype]
            display_categorization_results(valid_categories)
    
    with tab2:
        # Settings Tab (from version 2)
        st.write("### Confidence Configuration")
        configure_confidence_thresholds() # Function from version 2
        
        st.write("### Document Types Configuration")
        configure_document_types() # Function from version 2
        
        st.write("### Confidence Validation")
        with st.expander("Validate Confidence Model", expanded=False):
             validate_confidence_with_examples() # Function from version 2

# --- Helper Functions (Merged/Adapted from v2 and v3) ---

def configure_confidence_thresholds(): # From version 2
    """UI for configuring confidence thresholds."""
    st.session_state.confidence_thresholds["auto_accept"] = st.slider(
        "Auto-Accept Threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.confidence_thresholds.get("auto_accept", 0.85),
        step=0.05,
        key="auto_accept_slider_settings",
        help="Results above this threshold are considered highly reliable and can be automatically accepted"
    )
    st.session_state.confidence_thresholds["verification"] = st.slider(
        "Verification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.confidence_thresholds.get("verification", 0.6),
        step=0.05,
        key="verification_slider_settings",
        help="Results below this threshold require manual verification"
    )
    st.session_state.confidence_thresholds["rejection"] = st.slider(
        "Rejection Threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.confidence_thresholds.get("rejection", 0.4),
        step=0.05,
        key="rejection_slider_settings",
        help="Results below this threshold are considered unreliable and should be rejected or recategorized"
    )
    st.markdown("""
    **Threshold Explanation:**
    *   **Auto-Accept:** Highly reliable, automatically accepted.
    *   **Verification:** Requires manual verification.
    *   **Rejection:** Unreliable, should be rejected or recategorized.
    """)

def configure_document_types(): # From version 2
    """UI for configuring user-defined document types with descriptions."""
    st.write("Define custom document types and their descriptions for categorization:")
    
    # Ensure document_types exists and is a list of dicts
    if "document_types" not in st.session_state or not isinstance(st.session_state.document_types, list):
        # Re-initialize if structure is wrong
        st.session_state.document_types = [
            {"name": "Sales Contract", "description": "Contracts related to sales agreements and terms."},
            {"name": "Invoices", "description": "Billing documents..."},
            # ... (add other defaults) ...
            {"name": "Other", "description": "Any document not fitting..."}
        ]
        logger.warning("Document types state was missing or invalid, reset to default structure.")

    # Display current document types with descriptions and delete buttons
    indices_to_delete = []
    existing_names = {dtype.get("name", "").lower() for dtype in st.session_state.document_types}
    
    for i, doc_type_dict in enumerate(st.session_state.document_types):
        is_other_type = doc_type_dict.get("name") == "Other"
        
        with st.container():
            st.markdown(f"**Document Type {i+1}**")
            col1_dtype, col2_dtype, col3_dtype = st.columns([2, 3, 1])
            with col1_dtype:
                current_name = doc_type_dict.get("name", "")
                new_name = st.text_input(
                    f"Name {i+1}", 
                    value=current_name, 
                    key=f"doc_type_name_{i}", 
                    disabled=is_other_type, 
                    help="The name of the document category (must be unique)."
                )
                if new_name != current_name and not is_other_type:
                    if new_name.strip().lower() in existing_names - {current_name.lower()}:
                        st.error(f"Name \'{new_name}\' already exists. Please use a unique name.")
                    elif not new_name.strip():
                         st.error("Name cannot be empty.")
                    else:
                        st.session_state.document_types[i]["name"] = new_name.strip()
                        existing_names.remove(current_name.lower())
                        existing_names.add(new_name.strip().lower())
                        st.rerun() # Rerun to reflect name change immediately
                        
            with col2_dtype:
                current_desc = doc_type_dict.get("description", "")
                new_desc = st.text_area(
                    f"Description {i+1}", 
                    value=current_desc, 
                    key=f"doc_type_desc_{i}", 
                    height=50, 
                    help="A brief description of the document category."
                )
                if new_desc != current_desc:
                    st.session_state.document_types[i]["description"] = new_desc
                    # No rerun needed for description change

            with col3_dtype:
                if not is_other_type:
                    if st.button("Delete", key=f"delete_doc_type_{i}", use_container_width=True):
                        indices_to_delete.append(i)
                else:
                    st.write("_(Cannot delete \'Other\')_")
            st.markdown("--- ")

    # Process deletions
    if indices_to_delete:
        # Sort indices in reverse order to avoid index shifting issues
        indices_to_delete.sort(reverse=True)
        for index in indices_to_delete:
            if 0 <= index < len(st.session_state.document_types):
                del st.session_state.document_types[index]
        st.rerun()

    # Add new document type
    if st.button("Add New Document Type", key="add_doc_type_button"):
        # Find a unique default name
        default_new_name = "New Category"
        counter = 1
        while default_new_name.lower() in existing_names:
            counter += 1
            default_new_name = f"New Category {counter}"
            
        st.session_state.document_types.append({"name": default_new_name, "description": ""})
        st.rerun()

def validate_confidence_with_examples(): # From version 2
    """UI for validating confidence model with example documents."""
    st.write("Select example documents to test the confidence model:")
    # Placeholder for file selection and validation logic
    example_file = st.file_uploader("Upload an example document (optional)", type=["pdf", "docx", "txt"])
    if example_file:
        st.write(f"Selected: {example_file.name}")
        if st.button("Validate Confidence", key="validate_confidence_button"):
            st.info("Confidence validation logic not fully implemented in this example.")
            # Placeholder: Call categorization and display confidence scores
            # result = categorize_document(example_file_id, selected_model)
            # multi_factor = calculate_multi_factor_confidence(...)
            # calibrated = apply_confidence_calibration(...)
            # st.write(f"AI Confidence: {result[\"confidence\"]:.2f}")
            # st.write(f"Multi-Factor Confidence: {multi_factor[\"overall\"]:.2f}")
            # st.write(f"Calibrated Confidence: {calibrated:.2f}")

def categorize_document(file_id: str, model: str, valid_categories: List[str]) -> Dict[str, Any]: # Adapted for dynamic categories
    """
    Categorize a single document using Box AI /ai/ask endpoint.
    """
    client = st.session_state.client
    categories_string = ", ".join([f'"{cat}"' for cat in valid_categories])
    prompt = (f"""
    Analyze the provided document and determine its primary category.
    Respond ONLY with a JSON object containing the following keys:
    - "document_type": The most likely category from the list: [{categories_string}]
    - "confidence": A numerical score between 0.0 and 1.0 indicating your confidence in the assigned category.
    - "reasoning": A brief explanation (1-2 sentences) supporting your category choice.
    
    Example Response:
    {{
      "document_type": "Invoices",
      "confidence": 0.95,
      "reasoning": "The document contains invoice numbers, line items, and payment terms typical of an invoice."
    }}
    """)
    items = [{"type": "file", "id": file_id}]
    ai_config = {"mode": "completion"}
    try:
        logger.info(f"Sending categorization request for file {file_id} using model {model} with categories: {valid_categories}")
        response = client.ai.ask(model=model, prompt=prompt, items=items, ai_config=ai_config)
        logger.info(f"Received response for file {file_id}: {response.answer}")
        match = re.search(r"\{\s*.*\s*\}", response.answer, re.DOTALL)
        if not match:
            logger.error(f"Could not find valid JSON in response for file {file_id}: {response.answer}")
            raise ValueError("AI response did not contain a valid JSON object.")
        json_string = match.group(0)
        try:
            result_json = json.loads(json_string)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response for file {file_id}: {json_string} - Error: {e}")
            raise ValueError(f"AI response contained invalid JSON: {e}")
        if not all(key in result_json for key in ["document_type", "confidence", "reasoning"]):
            logger.error(f"JSON response missing required keys for file {file_id}: {result_json}")
            raise ValueError("AI response JSON is missing required keys (document_type, confidence, reasoning).")
        confidence = result_json.get("confidence")
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            logger.warning(f"Invalid confidence score \t{confidence}\t for file {file_id}. Setting to 0.0.")
            result_json["confidence"] = 0.0
        if result_json.get("document_type") not in valid_categories:
            logger.warning(f"AI returned invalid document type \t{result_json.get("document_type")}\t for file {file_id} (not in {valid_categories}). Setting to \'Other\'.")
            result_json["document_type"] = "Other"
        return result_json
    except Exception as e:
        logger.error(f"Error during AI categorization for file {file_id}: {e}")
        raise

def categorize_document_detailed(file_id: str, model: str, initial_category: str, valid_categories: List[str]) -> Dict[str, Any]: # Adapted for dynamic categories
    """
    Perform a more detailed categorization analysis.
    """
    client = st.session_state.client
    categories_string = ", ".join([f'"{cat}"' for cat in valid_categories])
    prompt = (f"""
    The initial analysis suggested the document might be a \t{initial_category}\t, but with low confidence.
    Please perform a more detailed analysis of the document content.
    Consider specific keywords, structure, and context.
    Respond ONLY with a JSON object containing:
    - "document_type": The most likely category from the list: [{categories_string}]
    - "confidence": Your confidence score (0.0 to 1.0) based on this detailed analysis.
    - "reasoning": A more detailed explanation (2-3 sentences) justifying your choice based on specific document elements.
    """)
    items = [{"type": "file", "id": file_id}]
    ai_config = {"mode": "completion"}
    try:
        logger.info(f"Sending detailed categorization request for file {file_id} using model {model} with categories: {valid_categories}")
        response = client.ai.ask(model=model, prompt=prompt, items=items, ai_config=ai_config)
        logger.info(f"Received detailed response for file {file_id}: {response.answer}")
        match = re.search(r"\{\s*.*\s*\}", response.answer, re.DOTALL)
        if not match:
            logger.error(f"Could not find valid JSON in detailed response for file {file_id}: {response.answer}")
            raise ValueError("AI detailed response did not contain a valid JSON object.")
        json_string = match.group(0)
        try:
            result_json = json.loads(json_string)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse detailed JSON response for file {file_id}: {json_string} - Error: {e}")
            raise ValueError(f"AI detailed response contained invalid JSON: {e}")
        if not all(key in result_json for key in ["document_type", "confidence", "reasoning"]):
            logger.error(f"Detailed JSON response missing required keys for file {file_id}: {result_json}")
            raise ValueError("AI detailed response JSON is missing required keys.")
        confidence = result_json.get("confidence")
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            logger.warning(f"Invalid detailed confidence score \t{confidence}\t for file {file_id}. Setting to 0.0.")
            result_json["confidence"] = 0.0
        if result_json.get("document_type") not in valid_categories:
            logger.warning(f"AI returned invalid detailed document type \t{result_json.get("document_type")}\t for file {file_id} (not in {valid_categories}). Setting to \'Other\'.")
            result_json["document_type"] = "Other"
        return result_json
    except Exception as e:
        logger.error(f"Error during detailed AI categorization for file {file_id}: {e}")
        raise

def combine_categorization_results(results: List[Dict[str, Any]]) -> Dict[str, Any]: # From version 2
    """Combine results from multiple models using weighted voting."""
    if not results:
        return {"document_type": "Other", "confidence": 0.0, "reasoning": "No results to combine"}
    category_votes = {}
    total_confidence = 0.0
    combined_reasoning = []
    for result in results:
        doc_type = result.get("document_type", "Other")
        confidence = result.get("confidence", 0.0)
        reasoning = result.get("reasoning", "")
        if doc_type not in category_votes:
            category_votes[doc_type] = 0.0
        category_votes[doc_type] += confidence
        total_confidence += confidence
        combined_reasoning.append(f"- Model suggested: {doc_type} (Conf: {confidence:.2f}): {reasoning}")
    if not category_votes:
        final_category = "Other"
        final_confidence = 0.0
    else:
        final_category = max(category_votes, key=category_votes.get)
        winning_confidences = [r["confidence"] for r in results if r.get("document_type") == final_category]
        final_confidence = sum(winning_confidences) / len(winning_confidences) if winning_confidences else 0.0
    return {
        "document_type": final_category,
        "confidence": final_confidence,
        "reasoning": "\n".join(combined_reasoning)
    }

def extract_document_features(file_id: str) -> Dict[str, Any]: # From version 2
    """Extract basic features from the document (placeholder)."""
    return {
        "text_length": 1500, 
        "contains_table": True, 
        "contains_signature": False 
    }

def calculate_multi_factor_confidence(ai_confidence: float, 
                                      features: Dict[str, Any], 
                                      predicted_type: str, 
                                      reasoning: str, 
                                      valid_categories: List[str]) -> Dict[str, float]: # Adapted for dynamic categories
    """Calculate a multi-factor confidence score."""
    weights = {"ai_confidence": 0.6, "feature_score": 0.3, "reasoning_score": 0.1}
    feature_score = 0.5
    if predicted_type == "Invoices" and features.get("contains_table"): feature_score = 0.8
    elif predicted_type == "Sales Contract" and features.get("contains_signature"): feature_score = 0.7
    reasoning_score = 0.5
    if len(reasoning) > 50: reasoning_score = 0.7
    overall_confidence = (
        ai_confidence * weights["ai_confidence"] +
        feature_score * weights["feature_score"] +
        reasoning_score * weights["reasoning_score"]
    )
    overall_confidence = max(0.0, min(1.0, overall_confidence))
    return {
        "overall": overall_confidence,
        "ai_confidence": ai_confidence,
        "feature_score": feature_score,
        "reasoning_score": reasoning_score
    }

def apply_confidence_calibration(document_type: str, confidence: float) -> float: # From version 2
    """Apply a simple calibration adjustment based on document type (placeholder)."""
    calibration_factors = {"Invoices": 1.05, "PII": 0.95}
    factor = calibration_factors.get(document_type, 1.0)
    calibrated_confidence = confidence * factor
    return max(0.0, min(1.0, calibrated_confidence))

def apply_confidence_thresholds(results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]: # From version 2
    """Apply status based on confidence thresholds."""
    thresholds = st.session_state.confidence_thresholds
    for file_id, result in results.items():
        confidence = result.get("calibrated_confidence", 0.0)
        if confidence >= thresholds["auto_accept"]:
            result["status"] = "Auto-Accepted"
            result["status_icon"] = "✅"
        elif confidence >= thresholds["verification"]:
            result["status"] = "Needs Verification"
            result["status_icon"] = "⚠️"
        else:
            result["status"] = "Rejected (Low Confidence)"
            result["status_icon"] = "❌"
    return results

def display_categorization_results(valid_categories: List[str]): # Adapted for dynamic categories
    """Display the results of the document categorization process."""
    st.write("### Categorization Results")
    results = st.session_state.document_categorization.get("results", {})
    errors = st.session_state.document_categorization.get("errors", {})
    if not results and not errors:
        st.info("No categorization results to display.")
        return
    
    # Prepare data including errors
    data = []
    for file_id, result in results.items():
        data.append({
            "file_id": file_id,
            "file_name": result.get("file_name", "Unknown"),
            "document_type": result.get("document_type", "Error"),
            "confidence": result.get("calibrated_confidence", 0.0),
            "status": result.get("status", "Unknown"),
            "status_icon": result.get("status_icon", "❓"),
            "reasoning": result.get("reasoning", "N/A"),
            "multi_factor_confidence": result.get("multi_factor_confidence", {}),
            "first_stage_type": result.get("first_stage_type"),
            "first_stage_confidence": result.get("first_stage_confidence")
        })
    for file_id, error_info in errors.items():
         data.append({
            "file_id": file_id,
            "file_name": error_info.get("file_name", "Unknown"),
            "document_type": "Error",
            "confidence": 0.0,
            "status": "Error",
            "status_icon": "🔥",
            "reasoning": error_info.get("error", "Unknown error"),
            "multi_factor_confidence": {},
            "first_stage_type": None,
            "first_stage_confidence": None
        })

    view_tab1, view_tab2 = st.tabs(["Table View", "Detailed View"])
    with view_tab1:
        st.write("#### Summary Table")
        if data:
            df = pd.DataFrame(data)
            display_df = df[["status_icon", "file_name", "document_type", "confidence", "status"]].rename(
                columns={
                    "status_icon": "", "file_name": "File Name", "document_type": "Predicted Category",
                    "confidence": "Confidence", "status": "Status"
                }
            )
            display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x:.1%}")
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No data to display in table view.")
            
    with view_tab2:
        st.write("#### Detailed Results and Feedback")
        if not data:
            st.info("No data to display in detailed view.")
            return
        thresholds = st.session_state.confidence_thresholds # Get thresholds for gauge
        for item in data:
            file_id = item["file_id"]
            file_name = item["file_name"]
            doc_type = item["document_type"]
            confidence = item["confidence"]
            status = item["status"]
            status_icon = item["status_icon"]
            reasoning = item["reasoning"]
            multi_factor = item["multi_factor_confidence"]
            
            st.markdown(f"**{status_icon} {file_name}**")
            col1_detail, col2_detail = st.columns([3, 1])
            with col1_detail:
                st.write(f"**Predicted Category:** {doc_type}")
                st.write(f"**Status:** {status}")
                if multi_factor and "overall" in multi_factor:
                    st.write(f"**Overall Confidence:** {confidence:.2f}")
                    with st.expander("Confidence Details", expanded=False):
                        st.write(f"- AI Confidence: {multi_factor.get('ai_confidence', 'N/A'):.2f}")
                        st.write(f"- Feature Score: {multi_factor.get('feature_score', 'N/A'):.2f}")
                        st.write(f"- Reasoning Score: {multi_factor.get('reasoning_score', 'N/A'):.2f}")
                else:
                    st.write(f"**Confidence:** {confidence:.2f}")
                if item.get("first_stage_type"):
                    st.write(f"_Initial category (low confidence): {item['first_stage_type']} ({item['first_stage_confidence']:.2f})_")
                with st.expander("AI Reasoning", expanded=False):
                    st.write(reasoning)
            with col2_detail:
                gauge_data = pd.DataFrame([{"confidence": confidence}])
                base = alt.Chart(gauge_data).encode(theta=alt.Theta("confidence", stack=True, scale=alt.Scale(domain=[0, 1])))
                color_scale = alt.Scale(
                    domain=[0, thresholds["rejection"], thresholds["verification"], thresholds["auto_accept"], 1],
                    range=["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c", "#5cb85c"]
                )
                arc = base.mark_arc(outerRadius=80, innerRadius=60).encode(color=alt.Color("confidence", scale=color_scale, legend=None))
                text = base.mark_text(radius=0, fontSize=20, align='center', baseline='middle').encode(
                    text=alt.Text("confidence", format=".1%"), color=alt.value("black")
                )
                gauge = arc + text
                st.altair_chart(gauge, use_container_width=True)
            
            # Add feedback collection form
            with st.expander("Provide Feedback / Correct Category", expanded=False):
                 # Pass dynamic category names
                collect_user_feedback(file_id, item, valid_categories)
            st.markdown("--- ")

def collect_user_feedback(file_id, result, valid_categories: List[str]): # Adapted for dynamic categories
    """Collect user feedback on categorization results."""
    if "categorization_feedback" not in st.session_state:
        st.session_state.categorization_feedback = {}
    
    document_types = valid_categories # Use the dynamic list
    col1_fb, col2_fb = st.columns(2)
    with col1_fb:
        current_type = result.get("document_type", "Other")
        try:
            default_index = document_types.index(current_type) if current_type in document_types else 0
        except ValueError:
             default_index = 0
        correct_category = st.selectbox(
            "Correct Category", options=document_types, index=default_index, key=f"feedback_category_{file_id}"
        )
    with col2_fb:
        confidence_rating = st.select_slider(
            "How confident are you in this categorization?",
            options=["Not at all", "Slightly", "Moderately", "Very", "Extremely"],
            value="Moderately", key=f"feedback_confidence_{file_id}"
        )
    feedback_text = st.text_area("Additional Feedback", key=f"feedback_text_{file_id}")
    if st.button("Submit Feedback", key=f"submit_feedback_{file_id}"):
        feedback_data = {
            "file_id": file_id,
            "predicted_category": result["document_type"],
            "predicted_confidence": result["confidence"],
            "correct_category": correct_category,
            "user_confidence_rating": confidence_rating,
            "feedback_text": feedback_text,
            "timestamp": datetime.datetime.now().isoformat()
        }
        st.session_state.categorization_feedback[file_id] = feedback_data
        logger.info(f"Feedback submitted for file {file_id}: {feedback_data}")
        st.success("Feedback submitted successfully!")

# Ensure this file can be imported without running the Streamlit app directly
if __name__ == "__main__":
    pass
