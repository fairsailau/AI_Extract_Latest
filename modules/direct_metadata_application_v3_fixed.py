# Merged version of direct_metadata_application_v3_fixed.py
# Incorporates bug fixes (strict type conversion, confidence filtering assurance)
# while preserving the original structure and UI elements.

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
    # --- This function is identical in both versions, keep as is --- 
    cache_key = f'{scope_str}_{template_key}'
    if cache_key in st.session_state.template_schema_cache:
        logger.info(f"Using cached schema for {scope_str}/{template_key}")
        return st.session_state.template_schema_cache[cache_key]

    try:
        if scope_str.startswith('enterprise'):
            scope_param = 'enterprise'
        elif scope_str == 'global':
            scope_param = 'global'
        else:
            logger.error(f"Unknown scope format: {scope_str}. Cannot determine scope parameter for SDK v3.")
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
        logger.error(f"Box API Error fetching template schema for {scope_str}/{template_key}: {e}")
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

    Args:
        key (str): The metadata field key.
        value: The original value.
        field_type (str): The target field type ('string', 'float', 'date', 'enum', 'multiSelect').

    Returns:
        Converted value.
    Raises:
        ConversionError: If the value cannot be converted to the target type.
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

def apply_metadata_direct():
    """
    Direct approach to apply metadata to Box files with type conversion based on template schema.
    (Merged version: Uses original structure with fixes integrated)
    """
    st.title("Apply Metadata")
    
    # --- Keep original debug checkbox and client checks --- 
    debug_mode = st.sidebar.checkbox("Debug Session State", key="debug_checkbox")
    if debug_mode:
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
                 first_key = next(iter(st.session_state.processing_state["results"]))
                 st.sidebar.write(f"**First Processing Result ({first_key}):**")
                 st.sidebar.json(st.session_state.processing_state["results"][first_key])
    
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
    st.sidebar.write("🔍 RAW processing_state")
    st.sidebar.json(processing_state)
    
    results_map = processing_state.get("results", {})
    file_id_to_file_name = {str(f["id"]): f["name"] for f in st.session_state.get("selected_files", []) if isinstance(f, dict) and "id" in f}
    
    # --- Keep original template selection logic, but use improved parsing from fix --- 
    if "metadata_config" not in st.session_state or not st.session_state.metadata_config.get("use_template"):
        st.warning("Structured extraction with a template was not selected or configured. Cannot apply metadata using a template.")
        return
        
    template_id_full = st.session_state.metadata_config.get("template_id")
    if not template_id_full or '_' not in template_id_full:
        st.error(f"Invalid or missing template ID in configuration: {template_id_full}. Cannot apply metadata.")
        return
        
    try:
        # Use improved parsing from fix
        scope_parts = template_id_full.split('_')
        if len(scope_parts) < 2:
             raise ValueError("Template ID format incorrect, expected scope_templateKey")
        scope_str = scope_parts[0] # 'enterprise' or 'global'
        template_key = '_'.join(scope_parts[1:])
        logger.info(f"Applying metadata using template: Scope='{scope_str}', Key='{template_key}'")
        st.write(f"Applying metadata using template: **{template_key}** (Scope: {scope_str})")
    except Exception as e:
        st.error(f"Could not parse template scope and key from configuration ('{template_id_full}'): {e}")
        return

    st.write(f"Found {len(results_map)} files with extraction results to process.")

    # --- Keep original button and loop structure --- 
    if st.button("Apply Extracted Metadata to Files", key="apply_metadata_button"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        success_count = 0
        error_count = 0
        total_files = len(results_map)
        conversion_warnings = {} # Store warnings per file

        # Get template schema once before the loop
        template_schema = get_template_schema(client, scope_str, template_key)
        if template_schema is None:
            st.error(f"Could not retrieve template schema for {scope_str}/{template_key}. Aborting metadata application.")
            return
        if not template_schema:
            st.warning(f"Template schema for {scope_str}/{template_key} is empty. No fields to apply.")
            # Allow proceeding, but nothing will be applied

        for i, (file_id, metadata_values) in enumerate(results_map.items()):
            file_name = file_id_to_file_name.get(file_id, f"File ID {file_id}")
            status_text.text(f"Processing {file_name}... ({i+1}/{total_files})")
            
            if not isinstance(metadata_values, dict):
                 logger.error(f"Metadata for file {file_id} is not a dictionary: {type(metadata_values)}. Skipping.")
                 st.error(f"Invalid metadata format for {file_name}. Skipping.")
                 error_count += 1
                 continue
                 
            # --- Integrate Fixes Here --- 
            metadata_to_apply = {}
            current_file_conversion_errors = []
            
            # 1. Filter confidence fields (as in original, but ensure it happens)
            filtered_metadata = filter_confidence_fields(metadata_values)
            logger.debug(f"File ID {file_id}: Filtered metadata (no confidence): {filtered_metadata}")

            # 2. Iterate through TEMPLATE schema fields and convert
            if template_schema: # Only proceed if schema is not empty
                for key, field_type in template_schema.items():
                    if key in filtered_metadata:
                        value = filtered_metadata[key]
                        try:
                            # Use the strict conversion function
                            converted_value = convert_value_for_template(key, value, field_type)
                            if converted_value is not None:
                                metadata_to_apply[key] = converted_value
                            else:
                                logger.info(f"Value for key '{key}' is None. Skipping for file {file_id}.")
                        except ConversionError as e:
                            # Catch specific conversion errors (from fix)
                            error_msg = f"Conversion error for key '{key}' (expected type '{field_type}', value: {repr(value)}): {e}. Field skipped."
                            logger.warning(error_msg)
                            current_file_conversion_errors.append(error_msg)
                        except Exception as e:
                            error_msg = f"Unexpected error processing key '{key}' for file {file_id}: {e}. Field skipped."
                            logger.error(error_msg)
                            current_file_conversion_errors.append(error_msg)
                    else:
                        logger.info(f"Template field '{key}' not found in extracted metadata for file {file_id}. Skipping field.")
            
            # Store conversion warnings for this file
            if current_file_conversion_errors:
                conversion_warnings[file_id] = current_file_conversion_errors

            # 3. Apply metadata if payload is not empty
            if not metadata_to_apply:
                if current_file_conversion_errors:
                    warn_msg = f"Metadata application skipped for file {file_name}: No fields could be successfully converted. Errors: {'; '.join(current_file_conversion_errors)}"
                    st.warning(warn_msg)
                    logger.warning(warn_msg)
                    error_count += 1 # Count as error if nothing could be applied due to conversion issues
                else:
                    info_msg = f"No matching metadata fields found or all values were None for file {file_name}. Nothing to apply."
                    st.info(info_msg)
                    logger.info(info_msg)
                    # Don't count as success or error if simply no matching fields
            else:
                logger.info(f"Attempting to apply metadata to file {file_id}: {metadata_to_apply}")
                try:
                    # Try update first, then create (original logic)
                    updated_metadata = client.file(file_id).metadata(scope=scope_str, template=template_key).update(metadata_to_apply)
                    success_msg = f"Metadata updated successfully for {file_name}."
                    logger.info(success_msg)
                    if file_id in conversion_warnings:
                         st.success(f"{success_msg} (with conversion warnings)")
                    else:
                         st.success(success_msg)
                    success_count += 1
                         
                except exception.BoxAPIException as e:
                    if e.status == 404:
                        logger.info(f"Metadata instance not found for file {file_id}, template {template_key}. Creating new instance.")
                        try:
                            created_metadata = client.file(file_id).metadata(scope=scope_str, template=template_key).create(metadata_to_apply)
                            success_msg = f"Metadata created successfully for {file_name}."
                            logger.info(success_msg)
                            if file_id in conversion_warnings:
                                 st.success(f"{success_msg} (with conversion warnings)")
                            else:
                                 st.success(success_msg)
                            success_count += 1
                                 
                        except exception.BoxAPIException as create_e:
                            error_msg = f"Box API Error creating metadata for {file_name}: {create_e}"
                            logger.error(error_msg)
                            st.error(error_msg)
                            error_count += 1
                    else:
                        error_msg = f"Box API Error updating metadata for {file_name}: {e}"
                        logger.error(error_msg)
                        st.error(error_msg)
                        error_count += 1
                except Exception as e:
                    error_msg = f"Unexpected error applying metadata to file {file_id} ({file_name}): {e}"
                    logger.exception(error_msg)
                    st.error(error_msg)
                    error_count += 1
            
            # Update progress bar (original logic)
            progress_bar.progress((i + 1) / total_files)

        # Final status update (original logic, slightly enhanced for warnings)
        status_text.text("Metadata application process complete.")
        st.write("---")
        st.write(f"**Summary:**")
        st.write(f"- Successfully applied/updated metadata for {success_count} files.")
        if conversion_warnings:
             st.write(f"- Conversion warnings occurred for {len(conversion_warnings)} files (some fields may have been skipped). Check logs for details.")
        if error_count > 0:
            st.write(f"- Failed to apply metadata or encountered errors for {error_count} files (see errors/warnings above)." )
        else:
             st.write(f"- No application errors encountered.")

    # --- Keep original debug payload display --- 
    if st.sidebar.checkbox("Show Processed Metadata Payload (Debug)", key="debug_payload_checkbox"):
        st.sidebar.write("### Processed Metadata (Example First File)")
        if results_map:
            first_file_id = next(iter(results_map))
            first_metadata = results_map[first_file_id]
            if isinstance(first_metadata, dict):
                 filtered = filter_confidence_fields(first_metadata)
                 st.sidebar.json(filtered)
            else:
                 st.sidebar.write("Invalid metadata format for first file.")

