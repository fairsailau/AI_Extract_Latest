# Merged version of direct_metadata_application_v3_fixed.py
# Incorporates bug fixes (strict type conversion, confidence filtering assurance)
# Restores per-file template mapping logic based on document categorization.
# Preserves the original structure and UI elements.

import streamlit as st
import logging
import json
from boxsdk import Client, exception
from dateutil import parser
from datetime import timezone

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Cache for template schemas to avoid repeated API calls
if 'template_schema_cache' not in st.session_state:
    st.session_state.template_schema_cache = {}

# Define a custom exception for conversion errors (from the fix)
class ConversionError(ValueError):
    pass

def get_template_schema(client, scope_str, template_key):
    """
    Fetches the metadata template schema from Box API (compatible with SDK v3.x).
    Uses a cache to avoid redundant API calls.
    Handles scope parameter as string ('enterprise' or 'global').
    
    Args:
        client: Box client object
        scope_str (str): The scope of the template (e.g., 'enterprise_12345' or 'global')
        template_key (str): The key of the template (e.g., 'invoiceData')
        
    Returns:
        dict: A dictionary mapping field keys to their types, or None if error.
    """
    cache_key = f'{scope_str}_{template_key}'
    if cache_key in st.session_state.template_schema_cache:
        logger.info(f"Using cached schema for {scope_str}/{template_key}")
        return st.session_state.template_schema_cache[cache_key]

    try:
        # Determine the correct scope string ('enterprise' or 'global') for SDK v3
        # The scope_str passed in should already be just 'enterprise' or 'global' based on parsing
        if scope_str.startswith('enterprise'):
            scope_param = 'enterprise'
        elif scope_str == 'global':
            scope_param = 'global'
        else:
            # If the parsing failed earlier, this might catch it, but parsing should handle it.
            logger.error(f"Invalid scope format passed to get_template_schema: {scope_str}. Expected 'enterprise' or 'global'.")
            st.session_state.template_schema_cache[cache_key] = None
            return None

        logger.info(f"Fetching template schema for {scope_str}/{template_key} using scope parameter '{scope_param}'")
        template = client.metadata_template(scope_param, template_key).get()
        
        if template and hasattr(template, 'fields') and template.fields:
            schema_map = {field['key']: field['type'] for field in template.fields}
            st.session_state.template_schema_cache[cache_key] = schema_map
            logger.info(f"Successfully fetched and cached schema for {scope_str}/{template_key}")
            return schema_map
        else:
            logger.warning(f"Template {scope_str}/{template_key} found but has no fields or is invalid.")
            st.session_state.template_schema_cache[cache_key] = {}
            return {}
            
    except exception.BoxAPIException as e:
        # Log specific Box API errors
        logger.error(f"Box API Error fetching template schema for {scope_str}/{template_key}: Status={e.status}, Code={e.code}, Message={e.message}")
        st.session_state.template_schema_cache[cache_key] = None 
        return None
    except Exception as e:
        logger.exception(f"Unexpected error fetching template schema for {scope_str}/{template_key}: {e}")
        st.session_state.template_schema_cache[cache_key] = None
        return None

def convert_value_for_template(key, value, field_type):
    """
    Converts a metadata value to the type specified by the template field.
    Raises ConversionError if conversion fails. (Modified from original to be strict)
    """
    # --- Use the strict version from the fix --- 
    if value is None:
        return None 
        
    original_value_repr = repr(value) # For logging

    try:
        if field_type == 'float':
            if isinstance(value, str):
                cleaned_value = value.replace('$', '').replace(',', '')
                try:
                    return float(cleaned_value)
                except ValueError:
                    raise ConversionError(f"Could not convert string '{value}' to float for key '{key}'.")
            elif isinstance(value, (int, float)):
                return float(value)
            else:
                 raise ConversionError(f"Value {original_value_repr} for key '{key}' is not a string or number, cannot convert to float.")
                 
        elif field_type == 'date':
            if isinstance(value, str):
                try:
                    dt = parser.parse(value)
                    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                except (parser.ParserError, ValueError) as e:
                    raise ConversionError(f"Could not parse date string '{value}' for key '{key}': {e}.")
            else:
                raise ConversionError(f"Value {original_value_repr} for key '{key}' is not a string, cannot convert to date.")
                
        elif field_type == 'string' or field_type == 'enum':
            if not isinstance(value, str):
                logger.info(f"Converting value {original_value_repr} to string for key '{key}' (type {field_type}).")
            return str(value)
            
        elif field_type == 'multiSelect':
            if isinstance(value, list):
                converted_list = [str(item) for item in value]
                if converted_list != value:
                     logger.info(f"Converting items in list {original_value_repr} to string for key '{key}' (type multiSelect).")
                return converted_list
            elif isinstance(value, str):
                logger.info(f"Converting string value {original_value_repr} to list of strings for key '{key}' (type multiSelect).")
                return [value]
            else:
                logger.info(f"Converting value {original_value_repr} to list of strings for key '{key}' (type multiSelect).")
                return [str(value)]
                
        else:
            logger.warning(f"Unknown field type '{field_type}' for key '{key}'. Cannot convert value {original_value_repr}.")
            raise ConversionError(f"Unknown field type '{field_type}' for key '{key}'.")
            
    except ConversionError: # Re-raise specific error
        raise
    except Exception as e: # Catch unexpected errors during conversion
        logger.error(f"Unexpected error converting value {original_value_repr} for key '{key}' (type {field_type}): {e}.")
        raise ConversionError(f"Unexpected error converting value for key '{key}': {e}")

def fix_metadata_format(metadata_values):
    """
    Fix the metadata format by converting string representations of dictionaries
    to actual Python dictionaries.
    """
    # --- This function is identical in both versions, keep as is --- 
    formatted_metadata = {}
    for key, value in metadata_values.items():
        if isinstance(value, str) and value.startswith('{') and value.endswith('}'):
            try:
                json_compatible_str = value.replace("'", '"')
                parsed_value = json.loads(json_compatible_str)
                formatted_metadata[key] = parsed_value
            except json.JSONDecodeError:
                formatted_metadata[key] = value
        else:
            formatted_metadata[key] = value
    return formatted_metadata

def flatten_metadata_for_template(metadata_values):
    """
    Flatten the metadata structure if needed (e.g., extracting from 'answer').
    """
    # --- This function is identical in both versions, keep as is --- 
    flattened_metadata = {}
    if 'answer' in metadata_values and isinstance(metadata_values['answer'], dict):
        for key, value in metadata_values['answer'].items():
            flattened_metadata[key] = value
    else:
        flattened_metadata = metadata_values.copy()
        
    keys_to_remove = ['ai_agent_info', 'created_at', 'completion_reason', 'answer']
    for key in keys_to_remove:
        if key in flattened_metadata:
            del flattened_metadata[key]
    return flattened_metadata

def filter_confidence_fields(metadata_values):
    """
    Filter out confidence score fields (keys ending with "_confidence").
    """
    # --- This function is identical in both versions, keep as is --- 
    return {key: value for key, value in metadata_values.items() if not key.endswith("_confidence")} 

def parse_template_id(template_id_full):
    """ Parses 'scope_templateKey' into (scope, templateKey). """
    if not template_id_full or '_' not in template_id_full:
        raise ValueError(f"Invalid template ID format: {template_id_full}")
    scope_parts = template_id_full.split('_')
    if len(scope_parts) < 2:
         raise ValueError(f"Template ID format incorrect, expected scope_templateKey: {template_id_full}")
    scope_str = scope_parts[0] # 'enterprise' or 'global'
    template_key = '_'.join(scope_parts[1:])
    # Validate scope format for API call
    if not scope_str.startswith('enterprise') and scope_str != 'global':
         raise ValueError(f"Invalid scope in template ID: {scope_str}")
    # Return the scope string needed for API calls ('enterprise' or 'global')
    api_scope = 'enterprise' if scope_str.startswith('enterprise') else 'global'
    return api_scope, template_key

def apply_metadata_to_file_direct(client, file_id, file_name, metadata_values, scope_str, template_key):
    """
    Applies metadata to a single file, handling type conversion and errors.
    (Internal function called by apply_metadata_direct)
    """
    logger.info(f"Starting metadata application for file ID {file_id} ({file_name}) with template {scope_str}/{template_key}")
    
    try:
        # 1. Pre-process metadata: Flatten (if needed) and filter confidence fields
        # flattened_values = flatten_metadata_for_template(metadata_values) # Apply if structure requires it
        filtered_metadata = filter_confidence_fields(metadata_values)
        logger.debug(f"File ID {file_id}: Filtered metadata (no confidence): {filtered_metadata}")

        # 2. Get template schema (using the specific scope/key for this file)
        template_schema = get_template_schema(client, scope_str, template_key)
        if template_schema is None:
            # Error already logged by get_template_schema
            error_msg = f"Could not retrieve template schema for {scope_str}/{template_key}. Cannot apply metadata to file {file_id} ({file_name})."
            st.error(f"Schema retrieval failed for {template_key}. Cannot apply metadata to {file_name}.")
            return False, error_msg
        if not template_schema: # Empty schema
             logger.warning(f"Template schema for {scope_str}/{template_key} is empty. No fields to apply for file {file_id} ({file_name}).")
             return True, "Template schema is empty, nothing to apply."

        # 3. Prepare metadata payload based on schema
        metadata_to_apply = {}
        conversion_errors = []
        
        for key, field_type in template_schema.items():
            if key in filtered_metadata:
                value = filtered_metadata[key]
                try:
                    converted_value = convert_value_for_template(key, value, field_type)
                    if converted_value is not None:
                        metadata_to_apply[key] = converted_value
                    else:
                        logger.info(f"Value for key '{key}' is None. Skipping for file {file_id}.")
                except ConversionError as e:
                    error_msg = f"Conversion error for key '{key}' (expected type '{field_type}', value: {repr(value)}): {e}. Field skipped."
                    logger.warning(error_msg)
                    conversion_errors.append(error_msg)
                except Exception as e:
                    error_msg = f"Unexpected error processing key '{key}' for file {file_id}: {e}. Field skipped."
                    logger.error(error_msg)
                    conversion_errors.append(error_msg)
            else:
                logger.info(f"Template field '{key}' not found in extracted metadata for file {file_id}. Skipping field.")

        # 4. Check if there's anything to apply
        if not metadata_to_apply:
            if conversion_errors:
                warn_msg = f"Metadata application skipped for file {file_name}: No fields could be successfully converted. Errors: {'; '.join(conversion_errors)}"
                st.warning(warn_msg)
                logger.warning(warn_msg)
                return False, f"No valid metadata fields to apply after conversion errors: {'; '.join(conversion_errors)}"
            else:
                info_msg = f"No matching metadata fields found or all values were None for file {file_name}. Nothing to apply."
                st.info(info_msg)
                logger.info(info_msg)
                return True, "No matching fields to apply"

        # 5. Apply metadata via Box API
        logger.info(f"Attempting to apply metadata to file {file_id}: {metadata_to_apply}")
        try:
            # Use the correct scope_str ('enterprise' or 'global') for the API call
            updated_metadata = client.file(file_id).metadata(scope=scope_str, template=template_key).update(metadata_to_apply)
            success_msg = f"Metadata updated successfully for {file_name}."
            logger.info(success_msg)
            if conversion_errors:
                 st.success(f"{success_msg} Warnings during conversion: {'; '.join(conversion_errors)}")
                 return True, f"{success_msg} Conversion warnings: {'; '.join(conversion_errors)}"
            else:
                 st.success(success_msg)
                 return True, success_msg
                 
        except exception.BoxAPIException as e:
            if e.status == 404: # Metadata instance doesn't exist, so create it
                logger.info(f"Metadata instance not found for file {file_id}, template {template_key}. Creating new instance.")
                try:
                    created_metadata = client.file(file_id).metadata(scope=scope_str, template=template_key).create(metadata_to_apply)
                    success_msg = f"Metadata created successfully for {file_name}."
                    logger.info(success_msg)
                    if conversion_errors:
                         st.success(f"{success_msg} Warnings during conversion: {'; '.join(conversion_errors)}")
                         return True, f"{success_msg} Conversion warnings: {'; '.join(conversion_errors)}"
                    else:
                         st.success(success_msg)
                         return True, success_msg
                         
                except exception.BoxAPIException as create_e:
                    error_msg = f"Box API Error creating metadata for {file_name}: Status={create_e.status}, Code={create_e.code}, Message={create_e.message}"
                    logger.error(error_msg)
                    st.error(error_msg)
                    return False, f"{error_msg}. Conversion warnings: {'; '.join(conversion_errors)}" if conversion_errors else error_msg
            else: # Other API error during update
                error_msg = f"Box API Error updating metadata for {file_name}: Status={e.status}, Code={e.code}, Message={e.message}"
                logger.error(error_msg)
                st.error(error_msg)
                return False, f"{error_msg}. Conversion warnings: {'; '.join(conversion_errors)}" if conversion_errors else error_msg
                
    except Exception as e:
        # Catch-all for unexpected errors during the process for this file
        error_msg = f"Unexpected error applying metadata to file {file_id} ({file_name}): {e}"
        logger.exception(error_msg) # Log full traceback
        st.error(error_msg)
        return False, error_msg

def apply_metadata_direct():
    """
    Main Streamlit page function to apply metadata using the direct approach.
    Restored logic to use per-file template mapping based on categorization.
    """
    st.title("Apply Metadata")
    
    # --- Keep original debug checkbox and client checks --- 
    debug_mode = st.sidebar.checkbox("Debug Session State", key="debug_checkbox")
    if debug_mode:
        # ... (debug code remains the same)
        st.sidebar.write("### Session State Debug")
        st.sidebar.write("**Session State Keys:**")
        st.sidebar.write(list(st.session_state.keys()))
        if "client" in st.session_state:
            st.sidebar.write("**Client:** Available")
            try:
                user = st.session_state.client.user().get()
                st.sidebar.write(f"**Authenticated as:** {user.name}")
            except Exception as e:
                st.sidebar.write(f"**Client Error:** {str(e)}")
        else:
            st.sidebar.write("**Client:** Not available")
        if "processing_state" in st.session_state:
            st.sidebar.write("**Processing State Keys:**")
            st.sidebar.write(list(st.session_state.processing_state.keys()))
            if st.session_state.processing_state.get("results"):
                 try:
                     first_key = next(iter(st.session_state.processing_state["results"]))
                     st.sidebar.write(f"**First Processing Result ({first_key}):**")
                     st.sidebar.json(st.session_state.processing_state["results"][first_key])
                 except StopIteration:
                     st.sidebar.write("Processing results dictionary is empty.")
            else:
                 st.sidebar.write("No 'results' key in processing_state.")
        if "document_categorization" in st.session_state:
             st.sidebar.write("**Document Categorization:**")
             st.sidebar.json(st.session_state.document_categorization)
        if "document_type_to_template" in st.session_state:
             st.sidebar.write("**Document Type to Template Mapping:**")
             st.sidebar.json(st.session_state.document_type_to_template)
        if "metadata_config" in st.session_state:
             st.sidebar.write("**Metadata Config:**")
             st.sidebar.json(st.session_state.metadata_config)
             
    if 'client' not in st.session_state:
        st.error("Box client not found. Please authenticate first.")
        if st.button("Go to Authentication", key="go_to_auth_btn"):
            st.session_state.current_page = "Home"
            st.rerun()
        return
    client = st.session_state.client
    
    try:
        user = client.user().get()
        logger.info(f"Verified client authentication as {user.name}")
        st.success(f"Authenticated as {user.name}")
    except Exception as e:
        logger.error(f"Error verifying client: {str(e)}")
        st.error(f"Authentication error: {str(e)}. Please re-authenticate.")
        if st.button("Go to Authentication", key="go_to_auth_error_btn"):
            st.session_state.current_page = "Home"
            st.rerun()
        return
    
    if "processing_state" not in st.session_state or not st.session_state.processing_state.get("results"):
        st.warning("No processing results available. Please process files first.")
        if st.button("Go to Process Files", key="go_to_process_files_btn"):
            st.session_state.current_page = "Process Files"
            st.rerun()
        return
    
    processing_state = st.session_state.processing_state
    logger.info(f"Processing state keys: {list(processing_state.keys())}")
    # st.sidebar.write("🔍 RAW processing_state") # Keep original sidebar dump if needed
    # st.sidebar.json(processing_state)
    
    results_map = processing_state.get("results", {})
    file_id_to_file_name = {str(f["id"]): f["name"] for f in st.session_state.get("selected_files", []) if isinstance(f, dict) and "id" in f}
    
    # --- Determine Default Template (used if no mapping found) --- 
    default_template_id_full = None
    default_scope_str = None
    default_template_key = None
    has_default_template = False
    
    if "metadata_config" in st.session_state and st.session_state.metadata_config.get("use_template"):
        default_template_id_full = st.session_state.metadata_config.get("template_id")
        if default_template_id_full:
            try:
                default_scope_str, default_template_key = parse_template_id(default_template_id_full)
                has_default_template = True
                logger.info(f"Default template configured: Scope='{default_scope_str}', Key='{default_template_key}'")
            except ValueError as e:
                st.error(f"Invalid default template ID format in configuration ('{default_template_id_full}'): {e}")
                # Cannot proceed without a valid default if mappings fail
                return 
        else:
             logger.warning("Structured extraction selected, but no default template ID found in metadata_config.")
             # Decide if this is an error or if we should skip files without mappings
             # For now, let's allow proceeding but warn.
             st.warning("Structured extraction selected, but no default template configured. Files without specific mappings will be skipped.")
    else:
        st.warning("Structured extraction with a template was not selected or configured. Cannot apply metadata using a template.")
        return

    # --- Get Categorization and Mapping Info --- 
    categorization_results = st.session_state.get("document_categorization", {}).get("results", {})
    doc_type_to_template_map = st.session_state.get("document_type_to_template", {})
    has_categorization_info = bool(categorization_results)
    has_mapping_info = bool(doc_type_to_template_map)

    if has_categorization_info:
        logger.info(f"Found categorization results for {len(categorization_results)} files.")
    if has_mapping_info:
        logger.info(f"Found document type to template mappings: {doc_type_to_template_map}")

    st.write(f"Found {len(results_map)} files with extraction results to process.")

    # --- Keep original button and loop structure --- 
    if st.button("Apply Extracted Metadata to Files", key="apply_metadata_button"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        success_count = 0
        error_count = 0
        skipped_count = 0 # Count files skipped due to no template
        total_files = len(results_map)
        conversion_warnings = {} # Store warnings per file

        for i, (file_id, metadata_values) in enumerate(results_map.items()):
            file_name = file_id_to_file_name.get(file_id, f"File ID {file_id}")
            status_text.text(f"Processing {file_name}... ({i+1}/{total_files})")
            
            if not isinstance(metadata_values, dict):
                 logger.error(f"Metadata for file {file_id} is not a dictionary: {type(metadata_values)}. Skipping.")
                 st.error(f"Invalid metadata format for {file_name}. Skipping.")
                 error_count += 1
                 continue
                 
            # --- Determine Template for THIS file --- 
            file_scope_str = None
            file_template_key = None
            template_source = "Default"
            
            # 1. Check categorization results for this file_id
            if has_categorization_info and file_id in categorization_results:
                doc_type = categorization_results[file_id].get("document_type")
                if doc_type and has_mapping_info and doc_type in doc_type_to_template_map:
                    mapped_template_id = doc_type_to_template_map[doc_type]
                    if mapped_template_id:
                        try:
                            file_scope_str, file_template_key = parse_template_id(mapped_template_id)
                            template_source = f"Mapping for '{doc_type}'"
                            logger.info(f"Using mapped template for file {file_id} ({doc_type}): {file_scope_str}/{file_template_key}")
                        except ValueError as e:
                            logger.warning(f"Invalid template ID '{mapped_template_id}' mapped for doc type '{doc_type}'. Falling back to default. Error: {e}")
                    else:
                        logger.info(f"No template mapped for doc type '{doc_type}'. Falling back to default.")
                else:
                     logger.info(f"No document type found or no mapping exists for file {file_id}. Falling back to default.")
            else:
                 logger.info(f"No categorization result found for file {file_id}. Falling back to default.")

            # 2. Fallback to default if no specific template was found/valid
            if not file_template_key:
                if has_default_template:
                    file_scope_str = default_scope_str
                    file_template_key = default_template_key
                    template_source = "Default"
                    logger.info(f"Using default template for file {file_id}: {file_scope_str}/{file_template_key}")
                else:
                    # No specific mapping AND no valid default template
                    logger.warning(f"No specific template mapping and no valid default template configured. Skipping metadata application for file {file_id} ({file_name}).")
                    st.warning(f"Skipping {file_name}: No applicable metadata template found.")
                    skipped_count += 1
                    progress_bar.progress((i + 1) / total_files) # Update progress even if skipped
                    continue # Skip to the next file
            
            # Display which template is being used
            status_text.text(f"Applying metadata to {file_name} using template '{file_template_key}' ({template_source})... ({i+1}/{total_files})")

            # --- Call internal function with the determined scope/key --- 
            success, message = apply_metadata_to_file_direct(
                client,
                file_id,
                file_name,
                metadata_values, 
                file_scope_str, # Pass the specific scope for this file
                file_template_key # Pass the specific key for this file
            )
            
            if success:
                success_count += 1
                # Store conversion warnings if message contains them (crude check)
                if "Conversion warnings:" in message:
                    conversion_warnings[file_id] = message.split("Conversion warnings:")[1].strip()
            else:
                error_count += 1
                # Error already logged/shown by apply_metadata_to_file_direct
            
            # Update progress bar (original logic)
            progress_bar.progress((i + 1) / total_files)

        # Final status update (original logic, slightly enhanced for warnings/skipped)
        status_text.text("Metadata application process complete.")
        st.write("---")
        st.write(f"**Summary:**")
        st.write(f"- Successfully applied/updated metadata for {success_count} files.")
        if conversion_warnings:
             st.write(f"- Conversion warnings occurred for {len(conversion_warnings)} files (some fields may have been skipped). Check logs or messages above for details.")
        if skipped_count > 0:
             st.write(f"- Skipped applying metadata for {skipped_count} files due to no applicable template.")
        if error_count > 0:
            st.write(f"- Failed to apply metadata or encountered errors for {error_count} files (see errors/warnings above)." )
        elif skipped_count == 0:
             st.write(f"- No application errors encountered.")

    # --- Keep original debug payload display --- 
    if st.sidebar.checkbox("Show Processed Metadata Payload (Debug)", key="debug_payload_checkbox"):
        # ... (debug code remains the same)
        st.sidebar.write("### Processed Metadata (Example First File)")
        if results_map:
            try:
                first_file_id = next(iter(results_map))
                first_metadata = results_map[first_file_id]
                if isinstance(first_metadata, dict):
                     filtered = filter_confidence_fields(first_metadata)
                     st.sidebar.json(filtered)
                else:
                     st.sidebar.write("Invalid metadata format for first file.")
            except StopIteration:
                 st.sidebar.write("Results map is empty.")
        else:
             st.sidebar.write("No results map found.")

