import streamlit as st
import logging

logger = logging.getLogger(__name__)

# Define the workflow steps (centralized definition)
workflow_steps = [
    {
        "id": "authentication",
        "title": "Login",
        "page": "Home",
        "icon": "🔑" # Icon might be hard to fit nicely in chevrons
    },
    {
        "id": "file_browser",
        "title": "Select Files",
        "page": "File Browser",
        "icon": "📁"
    },
    {
        "id": "document_categorization",
        "title": "Categorize",
        "page": "Document Categorization",
        "icon": "🏷️"
    },
    {
        "id": "metadata_config",
        "title": "Configure",
        "page": "Metadata Configuration",
        "icon": "⚙️"
    },
    {
        "id": "process_files",
        "title": "Process",
        "page": "Process Files",
        "icon": "🔄"
    },
    {
        "id": "view_results",
        "title": "Review",
        "page": "View Results",
        "icon": "👁️"
    },
    {
        "id": "apply_metadata",
        "title": "Apply",
        "page": "Apply Metadata",
        "icon": "✅"
    }
]

def navigate_to_step(page_id):
    """Callback function to update the current page."""
    # Check if the target page is valid and allowed (i.e., not a future step)
    target_index = -1
    current_index = -1
    for i, step in enumerate(workflow_steps):
        if step["page"] == page_id:
            target_index = i
        if step["page"] == st.session_state.current_page:
            current_index = i
            
    if target_index != -1 and target_index <= current_index:
        logger.info(f"Navigating to step: {page_id}")
        st.session_state.current_page = page_id
        # Optional: Add logic here to reset state for future steps if needed
        # This part needs careful consideration based on application logic
        # For now, just navigate.
        st.rerun() # Rerun to reflect the page change
    else:
        logger.warning(f"Attempted to navigate to invalid step: {page_id}")

def display_horizontal_workflow(current_page_id: str):
    """
    Displays the horizontal workflow indicator using Salesforce-style chevrons.

    Args:
        current_page_id: The page ID of the current step (e.g., "Home", "File Browser").
    """
    
    # Find the index of the current step
    current_step_index = -1
    for i, step in enumerate(workflow_steps):
        if step["page"] == current_page_id:
            current_step_index = i
            break
            
    # Inject CSS for chevron styling
    # Adapted from various online CSS chevron examples
    # Note: Direct click handling on these HTML elements to trigger Streamlit callbacks is not straightforward.
    # We use st.button overlaid for interaction.
    css = """
    <style>
        .chevron-container {
            display: flex;
            justify-content: center; /* Center the chevrons */
            list-style: none;
            padding: 0;
            margin: 20px 0; /* Add some margin */
            width: 100%;
            overflow-x: auto; /* Allow horizontal scrolling if needed */
        }
        .chevron-step {
            background-color: #e9ecef; /* Default upcoming background */
            color: #6c757d; /* Default upcoming text */
            padding: 0.5rem 1rem 0.5rem 2rem; /* Adjust padding */
            margin-right: -1rem; /* Overlap chevrons */
            position: relative;
            text-align: center;
            min-width: 120px; /* Minimum width for each step */
            white-space: nowrap;
            border: 1px solid #ced4da;
            cursor: default; /* Default cursor */
        }
        .chevron-step::before, .chevron-step::after {
            content: "";
            position: absolute;
            top: 0;
            border: 0 solid transparent;
            border-width: 1.55rem 1rem; /* Controls size/angle of arrow */
            width: 0;
            height: 0;
        }
        .chevron-step::before {
            left: -0.05rem; /* Position left arrow */
            border-left-color: white; /* Match page background */
            border-left-width: 1rem;
        }
        .chevron-step::after {
            left: 100%;
            z-index: 2;
            border-left-color: #e9ecef; /* Match step background */
        }
        /* First step doesn't need the left cutout */
        .chevron-step:first-child {
            padding-left: 1rem;
            border-top-left-radius: 5px;
            border-bottom-left-radius: 5px;
        }
        .chevron-step:first-child::before {
            display: none;
        }
        /* Last step doesn't need the right arrow */
        .chevron-step:last-child {
            margin-right: 0;
            padding-right: 1rem;
            border-top-right-radius: 5px;
            border-bottom-right-radius: 5px;
        }
        .chevron-step:last-child::after {
            display: none;
        }

        /* Completed Step Styling */
        .chevron-step-completed {
            background-color: #cfe2ff; /* Light blue background */
            color: #052c65; /* Dark blue text */
            cursor: pointer; /* Make clickable */
            border-color: #9ec5fe;
        }
        .chevron-step-completed::after {
            border-left-color: #cfe2ff; /* Match completed background */
        }
        .chevron-step-completed:hover {
            background-color: #b6d4fe; /* Slightly darker blue on hover */
        }
        .chevron-step-completed:hover::after {
            border-left-color: #b6d4fe;
        }

        /* Current Step Styling */
        .chevron-step-current {
            background-color: #0d6efd; /* Blue background */
            color: white;
            font-weight: bold;
            z-index: 3; /* Ensure current step overlaps others */
            border-color: #0a58ca;
        }
        .chevron-step-current::after {
            border-left-color: #0d6efd; /* Match current background */
        }
        
        /* Link styling within chevrons */
        .chevron-step a {
            color: inherit; /* Inherit color from parent */
            text-decoration: none;
            display: block; /* Make link fill the step */
            padding: inherit; /* Use parent padding */
            margin: -0.5rem -1rem -0.5rem -2rem; /* Fill padding area */
        }
        .chevron-step:first-child a {
             margin-left: -1rem; /* Adjust for first step */
        }
        .chevron-step:last-child a {
             margin-right: -1rem; /* Adjust for last step */
        }

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    # Generate HTML for the chevrons
    html_content = "<div class=\"chevron-container\">"
    
    for i, step in enumerate(workflow_steps):
        # Determine CSS class based on status
        status_class = ""
        is_clickable = False
        if i < current_step_index:
            status_class = "chevron-step-completed"
            is_clickable = True
        elif i == current_step_index:
            status_class = "chevron-step-current"
            is_clickable = False # Current step not clickable to itself
        else:
            status_class = "chevron-step-upcoming"
            is_clickable = False

        step_html = f"<div class=\"chevron-step {status_class}\" "
        step_html += f" title=\"{step['title']}\">"
        
        # Use Streamlit buttons overlaid for click handling (more reliable than HTML links)
        # We place buttons in columns *after* rendering the visual chevrons.
        # This HTML is purely for visual representation.
        step_html += f"{step['title']}" # Display title
        # Add checkmark for completed steps? (Optional)
        if i < current_step_index:
             step_html += " ✓"
             
        step_html += "</div>"
        html_content += step_html
        
    html_content += "</div>"
    
    # Render the visual chevrons
    st.markdown(html_content, unsafe_allow_html=True)

    # --- Click Handling using overlaid st.button --- 
    # Create columns matching the number of steps
    cols = st.columns(len(workflow_steps))
    for i, step in enumerate(workflow_steps):
        with cols[i]:
            is_clickable = i < current_step_index # Only allow clicking on *previous* completed steps
            # Create a mostly invisible button to capture clicks
            # Styling these buttons perfectly over the chevrons is hard.
            # This provides functionality but might not align perfectly.
            st.button(
                label=" "* (i+1), # Use spaces for label, unique key needed
                key=f"nav_btn_{step['id']}",
                on_click=navigate_to_step,
                args=(step["page"],),
                disabled=not is_clickable,
                use_container_width=True,
                help=f"Go to {step['title']}" if is_clickable else step['title']
            )
            # Add a small visual cue below the button column to show what it represents
            # st.caption(f"{step['title']}") # This might clutter the UI


