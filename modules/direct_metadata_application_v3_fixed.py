# Merged version of direct_metadata_application_v3_fixed.py
# Incorporates bug fixes (strict type conversion, confidence filtering assurance)
# Restores per-file template mapping logic based on document categorization.
# Corrects template ID parsing for FULL scope and simple key.
# Corrects metadata update method using SDK operations pattern.
# Preserves the original structure and UI elements.
# ADDED: Fallback to global.properties if no custom template is found.

import streamlit as st
import logging
import json
from boxsdk import Client, exception
from boxsdk.object.metadata import MetadataUpdate # Import MetadataUpdate
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

def get_template_schema(client, full_scope, template_key):
    """
    Fetches the metadata template schema from Box API (compatible with SDK v3.x).
    Uses a cache to avoid redundant API calls.
    Uses FULL scope (e.g., enterprise_12345) and simple template key.
    
    Args:
        client: Box client object
        full_scope (str): The full scope identifier (e.g., "enterprise_12345" or "global").
        template_key (str): The key of the template (e.g., "homeLoan").
        
    Returns:
        dict: A dictionary mapping field keys to their types, or None if error.
    """
    # --- No changes needed here --- 
    cache_key = f'{full_scope}_{template_key}' 
    if cache_key in st.session_state.template_schema_cache:
        logger.info(f"Using cached schema for {full_scope}/{template_key}")
        return st.session_state.template_schema_cache[cache_key]

    try:
        logger.info(f"Fetching template schema for {full_scope}/{template_key}")
        template = client.metadata_template(full_scope, template_key).get()
        
        if template and hasattr(template, 'fields') and template.fields:
            schema_map = {field['key']: field['type'] for field in template.fields}
            st.session_state.template_schema_cache[cache_key] = schema_map
            logger.info(f"Successfully fetched and cached schema for {full_scope}/{template_key}")
            return schema_map
        else:
            logger.warning(f"Template {full_scope}/{template_key} found but has no fields or is invalid.")
            st.session_state.template_schema_cache[cache_key] = {}
            return {}
            
    except exception.BoxAPIException as e:
        logger.error(f"Box API Error fetching template schema for {full_scope}/{template_key}: Status={e.status}, Code={e.code}, Message={e.message}")
        st.session_state.template_schema_cache[cache_key] = None 
        return None
    except Exception as e:
        logger.exception(f"Unexpected error fetching template schema for {full_scope}/{template_key}: {e}")
        st.session_state.template_schema_cache[cache_key] = None
        return None

def convert_value_for_template(key, value, field_type):
    """
    Converts a metadata value to the type specified by the template field.
    Raises ConversionError if conversion fails. (Modified from original to be strict)
    """
    # --- No changes needed here --- 
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
    # --- No changes needed here --- 
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
    # --- No changes needed here --- 
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
    # --- No changes needed here --- 
    return {key: value for key, value in metadata_values.items() if not key.endswith("_confidence")} 

def parse_template_id(template_id_full):
    """ 
    Parses 'scope_templateKey' (e.g., 'enterprise_12345_myTemplate') into 
    (full_scope, template_key).
    Corrected based on user feedback.
    
    Args:
        template_id_full (str): The combined template ID string.
        
    Returns:
        tuple: (full_scope, template_key)
               full_scope (str): The full scope identifier (e.g., 'enterprise_12345' or 'global')
               template_key (str): The actual key of the template (e.g., 'myTemplate')
    Raises:
        ValueError: If the format is invalid.
    """
    # --- No changes needed here --- 
    if not template_id_full or '_' not in template_id_full:
        raise ValueError(f"Invalid template ID format: {template_id_full}")
    
    last_underscore_index = template_id_full.rfind('_')
    if last_underscore_index == 0 or last_underscore_index == len(template_id_full) - 1:
        raise ValueError(f"Template ID format incorrect, expected scope_templateKey: {template_id_full}")
        
    full_scope = template_id_full[:last_underscore_index]
    template_key = template_id_full[last_underscore_index + 1:]
    
    if not full_scope.startswith('enterprise_') and full_scope != 'global':
         if not full_scope == 'enterprise': 
            logger.warning(f"Scope format '{full_scope}' might be unexpected. Expected 'enterprise_...' or 'global'.")
        
    logger.debug(f"Parsed template ID '{template_id_full}' -> full_scope='{full_scope}', template_key='{template_key}'")
    return full_scope, template_key

def apply_metadata_to_file_direct(client, file_id, file_name, metadata_values, full_scope, template_key):
    """
    Applies metadata to a single file using the correct SDK update pattern.
    Uses FULL scope and simple template key.
    Handles both custom templates and the global.properties fallback.
    (Internal function called by apply_metadata_direct)
    
    Args:
        client: Box client object
        file_id (str): ID of the file.
        file_name (str): Name of the file.
        metadata_values (dict): Extracted metadata (before filtering/conversion).
        full_scope (str): The full scope identifier (e.g., 'enterprise_12345' or 'global').
        template_key (str): The actual template key (e.g., 'homeLoan' or 'properties').
        
    Returns:
        tuple: (success_flag, message_string)
    """
    logger.info(f"Starting metadata application for file ID {file_id} ({file_name}) with template {full_scope}/{template_key}")
    
    # --- ADDED: Check if using global.properties --- 
    is_global_properties = (full_scope == 'global' and template_key == 'properties')
    
    try:
        # 1. Pre-process metadata: Filter confidence fields
        filtered_metadata = filter_confidence_fields(metadata_values) 
        logger.debug(f"File ID {file_id}: Filtered metadata (no confidence): {filtered_metadata}")

        metadata_to_apply = {}
        conversion_errors = []

        # --- MODIFIED: Handle global.properties vs custom template --- 
        if is_global_properties:
            logger.info(f"Applying to global.properties for file {file_id}. Skipping schema validation and type conversion.")
            # For global.properties, just ensure basic types or convert to string
            for key, value in filtered_metadata.items():
                if value is None:
                    continue # Skip None values
                if isinstance(value, (str, int, float, bool)):
                    metadata_to_apply[key] = value
                else:
                    try:
                        metadata_to_apply[key] = str(value)
                        logger.info(f"Converted value for key '{key}' to string for global.properties.")
                    except Exception as str_e:
                        error_msg = f"Could not convert value for key '{key}' to string for global.properties: {str_e}. Field skipped."
                        logger.warning(error_msg)
                        conversion_errors.append(error_msg)
        else:
            # For custom templates, use schema validation and conversion
            # 2. Get template schema (using the correctly parsed full_scope/template_key)
            template_schema = get_template_schema(client, full_scope, template_key)
            if template_schema is None:
                error_msg = f"Could not retrieve template schema for {full_scope}/{template_key}. Cannot apply metadata to file {file_id} ({file_name})."
                st.error(f"Schema retrieval failed for template '{template_key}'. Cannot apply metadata to {file_name}. Check template key and permissions.")
                return False, error_msg
            if not template_schema: # Empty schema
                 logger.warning(f"Template schema for {full_scope}/{template_key} is empty. No fields to apply for file {file_id} ({file_name}).")
                 st.info(f"Template '{template_key}' has no fields. Nothing to apply to {file_name}.")
                 return True, "Template schema is empty, nothing to apply."

            # 3. Prepare metadata payload based on schema
            for key, field_type in template_schema.items():
                if key in filtered_metadata:
                    value = filtered_metadata[key]
                    try:
                        converted_value = convert_value_for_template(key, value, field_type)
                        if converted_value is not None:
                            metadata_to_apply[key] = converted_value
                        else:
                            logger.info(f"Value for key '{key}' is None after conversion. Skipping for file {file_id}.")
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

        # 4. Check if there's anything to apply (common check)
        if not metadata_to_apply:
            if conversion_errors:
                warn_msg = f"Metadata application skipped for file {file_name}: No fields could be successfully converted or prepared. Errors: {'; '.join(conversion_errors)}"
                st.warning(warn_msg)
                logger.warning(warn_msg)
                return False, f"No valid metadata fields to apply after conversion/preparation errors: {'; '.join(conversion_errors)}"
            else:
                info_msg = f"No matching metadata fields found or all values were None for file {file_name}. Nothing to apply."
                st.info(info_msg)
                logger.info(info_msg)
                return True, "No matching fields to apply"

        # 5. Apply metadata via Box API using the correct update pattern
        logger.info(f"Attempting to apply metadata to file {file_id} using template {full_scope}/{template_key} with payload: {metadata_to_apply}")
        try:
            metadata_instance = client.file(file_id).metadata(scope=full_scope, template=template_key)
            
            try:
                # Try to create first
                metadata = metadata_instance.create(metadata_to_apply)
                logger.info(f"Successfully CREATED metadata instance {full_scope}/{template_key} for file {file_id}.")
                message = f"Metadata applied successfully."
                if conversion_errors:
                    message += f" Conversion warnings: {'; '.join(conversion_errors)}"
                return True, message
                
            except exception.BoxAPIException as e:
                if e.status == 409 and e.code == 'instance_tuple_already_exists':
                    # Instance exists, so update instead
                    logger.info(f"Metadata instance {full_scope}/{template_key} already exists for file {file_id}. Attempting update.")
                    try:
                        # Use MetadataUpdate for robust updates
                        update = MetadataUpdate()
                        for key, value in metadata_to_apply.items():
                            # Check if field exists in schema (only for custom templates)
                            if not is_global_properties:
                                current_value = metadata_instance.get().get(key)
                                if current_value is None:
                                    update.add(f'/{key}', value)
                                else:
                                    update.replace(f'/{key}', value)
                            else:
                                # For global.properties, always add/replace
                                update.add(f'/{key}', value) # Add will replace if exists for global.properties
                        
                        # Apply the update
                        updated_metadata = metadata_instance.update(update)
                        logger.info(f"Successfully UPDATED metadata instance {full_scope}/{template_key} for file {file_id}.")
                        message = f"Metadata updated successfully."
                        if conversion_errors:
                            message += f" Conversion warnings: {'; '.join(conversion_errors)}"
                        return True, message
                        
                    except exception.BoxAPIException as update_e:
                        error_msg = f"Error UPDATING metadata instance {full_scope}/{template_key} for file {file_id}: Status={update_e.status}, Code={update_e.code}, Message={update_e.message}"
                        logger.error(error_msg)
                        st.error(f"Error updating metadata for {file_name}: {update_e.message}")
                        return False, error_msg
                    except Exception as update_e:
                        error_msg = f"Unexpected error UPDATING metadata instance {full_scope}/{template_key} for file {file_id}: {update_e}"
                        logger.exception(error_msg)
                        st.error(f"Unexpected error updating metadata for {file_name}: {update_e}")
                        return False, error_msg
                else:
                    # Other API error during create
                    error_msg = f"Error CREATING metadata instance {full_scope}/{template_key} for file {file_id}: Status={e.status}, Code={e.code}, Message={e.message}"
                    logger.error(error_msg)
                    st.error(f"Error applying metadata for {file_name}: {e.message}")
                    return False, error_msg
                    
        except Exception as e:
            error_msg = f"Unexpected error applying metadata instance {full_scope}/{template_key} for file {file_id}: {e}"
            logger.exception(error_msg)
            st.error(f"Unexpected error applying metadata for {file_name}: {e}")
            return False, error_msg
            
    except Exception as outer_e:
        # Catch errors in pre-processing steps
        error_msg = f"Unexpected error during pre-processing for file {file_id} ({file_name}): {outer_e}"
        logger.exception(error_msg)
        st.error(f"Error preparing metadata for {file_name}: {outer_e}")
        return False, error_msg

def apply_metadata_direct():
    """
    Main Streamlit page function for applying metadata.
    Uses the internal apply_metadata_to_file_direct function.
    Includes fallback to global.properties.
    """
    st.subheader("Step 7: Apply Metadata") # Changed title to subheader for consistency
    st.write("Apply the extracted metadata fields to your selected Box files.")
    
    # --- Authentication Check (same as before) --- 
    if 'client' not in st.session_state:
        st.error("Box client not found. Please authenticate first.")
        return
    client = st.session_state.client
    try:
        user = client.user().get()
        st.success(f"Authenticated as {user.name}")
    except Exception as e:
        st.error(f"Authentication error: {str(e)}. Please re-authenticate.")
        return

    # --- Check for Results (same as before) --- 
    if "processing_state" not in st.session_state or not st.session_state.processing_state.get("results"):
        st.warning("No processing results available. Please process files first.")
        return
        
    results_map = st.session_state.processing_state.get("results", {})
    
    # --- Get File Name Mapping (same as before) --- 
    file_id_to_file_name = {}
    if "selected_files" in st.session_state and st.session_state.selected_files:
        for file_info in st.session_state.selected_files:
            if isinstance(file_info, dict) and "id" in file_info:
                file_id_to_file_name[str(file_info["id"])] = file_info.get("name", f"File ID {file_info['id']}")
    else:
        # Fallback if selected_files isn't populated correctly
        for file_id in results_map.keys():
             if str(file_id) not in file_id_to_file_name:
                 file_id_to_file_name[str(file_id)] = f"File ID {file_id}"

    # --- Determine Default Template (same as before) --- 
    default_full_scope = None
    default_template_key = None
    has_default_template = False
    if "metadata_config" in st.session_state and st.session_state.metadata_config.get("use_template"):
        default_template_id_full = st.session_state.metadata_config.get("default_template_id")
        if default_template_id_full:
            try:
                default_full_scope, default_template_key = parse_template_id(default_template_id_full)
                has_default_template = True
                logger.info(f"Default template configured: {default_full_scope}/{default_template_key}")
            except ValueError as e:
                logger.warning(f"Invalid default template ID configured: {default_template_id_full}. Error: {e}")
                st.warning(f"The configured default template ID '{default_template_id_full}' is invalid. It will be ignored.")
        else:
             logger.info("Structured extraction selected, but no default template ID found in metadata_config.")
             # No warning needed here, handled per file later

    # --- Get Categorization and Mapping Info (same as before) --- 
    has_categorization_info = "document_categorization" in st.session_state and st.session_state.document_categorization.get("results")
    categorization_results = st.session_state.document_categorization.get("results", {}) if has_categorization_info else {}
    has_mapping_info = "document_type_to_template" in st.session_state
    doc_type_to_template_map = st.session_state.get("document_type_to_template", {}) if has_mapping_info else {}
    if has_categorization_info:
        logger.info(f"Found categorization results for {len(categorization_results)} files.")
    if has_mapping_info:
        logger.info(f"Found document type to template mappings: {doc_type_to_template_map}")

    st.write(f"Found {len(results_map)} files with extraction results to process.")

    # --- Apply Button and Loop --- 
    if st.button("Apply Extracted Metadata to Files", key="apply_metadata_button"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        success_count = 0
        error_count = 0
        skipped_count = 0 # Not used with fallback, but keep variable
        total_files = len(results_map)
        all_conversion_warnings = {} 

        for i, (file_id_raw, metadata_values) in enumerate(results_map.items()):
            file_id = str(file_id_raw) # Ensure string ID
            file_name = file_id_to_file_name.get(file_id, f"File ID {file_id}")
            status_text.text(f"Processing {file_name}... ({i+1}/{total_files})")
            
            if not isinstance(metadata_values, dict):
                 logger.error(f"Metadata for file {file_id} is not a dictionary: {type(metadata_values)}. Skipping.")
                 st.error(f"Invalid metadata format for {file_name}. Skipping.")
                 error_count += 1
                 progress_bar.progress((i + 1) / total_files)
                 continue
                 
            # --- MODIFIED: Determine Template or Fallback --- 
            file_full_scope = None
            file_template_key = None
            template_source = "Unknown"
            
            # 1. Check categorization mapping
            if has_categorization_info and file_id in categorization_results:
                doc_type = categorization_results[file_id].get("document_type")
                if doc_type and has_mapping_info and doc_type in doc_type_to_template_map:
                    mapped_template_id = doc_type_to_template_map[doc_type]
                    if mapped_template_id:
                        try:
                            file_full_scope, file_template_key = parse_template_id(mapped_template_id)
                            template_source = f"Mapping for '{doc_type}'"
                            logger.info(f"Using mapped template for file {file_id} ({doc_type}): {file_full_scope}/{file_template_key}")
                        except ValueError as e:
                            logger.warning(f"Invalid template ID '{mapped_template_id}' mapped for doc type '{doc_type}'. Falling back. Error: {e}")
                    else:
                        logger.info(f"No template mapped for doc type '{doc_type}'. Falling back.")
                else:
                     logger.info(f"No document type found or no mapping exists for file {file_id}. Falling back.")
            else:
                 logger.info(f"No categorization result found for file {file_id}. Falling back.")

            # 2. Fallback to default custom template if no specific one found
            if not file_template_key:
                if has_default_template:
                    file_full_scope = default_full_scope
                    file_template_key = default_template_key
                    template_source = "Default Custom"
                    logger.info(f"Using default custom template for file {file_id}: {file_full_scope}/{file_template_key}")
                else:
                    # 3. Fallback to global.properties if no custom template found
                    logger.info(f"No specific or default custom template found for file {file_id}. Falling back to global.properties.")
                    file_full_scope = "global"
                    file_template_key = "properties"
                    template_source = "Fallback (global.properties)"
                    # No 'continue' here, proceed with global.properties
            
            # Display which template is being used
            status_text.text(f"Applying metadata to {file_name} using template '{file_template_key}' ({template_source})... ({i+1}/{total_files})")

            # --- Call internal function with the determined full_scope/template_key --- 
            success, message = apply_metadata_to_file_direct(
                client,
                file_id,
                file_name,
                metadata_values, 
                file_full_scope,    # Pass the specific full_scope for this file
                file_template_key   # Pass the specific template_key for this file
            )
            
            if success:
                success_count += 1
                if "Conversion warnings:" in message:
                    warning_detail = message.split("Conversion warnings:")[1].strip()
                    if warning_detail:
                        all_conversion_warnings[file_id] = warning_detail
            else:
                error_count += 1
            
            progress_bar.progress((i + 1) / total_files)

        # --- Final Status Update (same as before, adjusted message) --- 
        status_text.text("Metadata application process complete.")
        st.write("---")
        st.write(f"**Summary:**")
        st.write(f"- Successfully applied/updated metadata for {success_count} files.")
        if all_conversion_warnings:
             st.write(f"- Conversion warnings occurred for {len(all_conversion_warnings)} files (some fields may have been skipped). Check logs or messages above for details.")
        # Removed skipped count message as fallback is used
        if error_count > 0:
            st.write(f"- Failed to apply metadata or encountered errors for {error_count} files (see errors/warnings above)." )
        elif not all_conversion_warnings:
             st.write(f"- No application errors or conversion warnings encountered.")

    # --- Debug Payload Display (same as before) --- 
    if st.sidebar.checkbox("Show Processed Metadata Payload (Debug)", key="debug_payload_checkbox"):
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


