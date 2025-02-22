from suite2p.parameters import DB, SETTINGS

def generate_markdown(settings):
    """Generates Markdown with optimized section creation for nested dictionaries."""
    md_content = "# Settings for Suite2p\n\n"
    md_content += ("Suite2p can be run with different configurations using the ops dictionary. "
                   "The ops dictionary will describe the settings used for a particular run of the pipeline. "
                   "Here is a summary of all the parameters that the pipeline takes and their default values.\n\n")

    # Collect non-nested entries for the top table
    flat_entries = []
    nested_entries = {}

    for key, value in settings.items():
        if isinstance(value, dict) and "gui_name" not in value:
            # Nested dictionary -> Create a subsection
            nested_entries[key] = value
        else:
            # Flat entry -> Add to top-level table
            flat_entries.append((key, value))

    # Generate the top-level table for standalone settings
    if flat_entries:
        md_content += "## General Settings\n\n"
        md_content += generate_table(flat_entries)

    # Function to process nested dictionaries
    def process_nested_dict(nested_dict, level=3):
        """Recursively process nested dictionaries into Markdown sections."""
        md = ""

        for key, value in nested_dict.items():
            section_title = key.replace("_", " ").capitalize()

            # Process each key in the nested dictionary
            if isinstance(value, dict):
                # If the nested item is a dictionary, generate the table for its values
                md += f"### {section_title}\n\n"
                md += generate_table_for_nested_dict(value)

        return md

    # Process nested sections
    md_content += process_nested_dict(nested_entries)

    return md_content


def generate_table(entries):
    """Generates a Markdown table for given dictionary entries."""
    table_md = "| Key | GUI Name | Type | Default | Description |\n"
    table_md += "|---|---|---|---|---|\n"

    for key, value in entries:
        # Ensure the value is a dictionary before trying to access its keys
        if isinstance(value, dict):
            table_md += f"| `{key}` | {value.get('gui_name', 'N/A')} | `{value.get('type', 'N/A')}` | `{value.get('default', 'N/A')}` | {value.get('description', 'N/A')} |\n"
        else:
            # Handle the case where the value is not a dictionary (e.g., string or other simple types)
            table_md += f"| `{key}` | N/A | N/A | `{value}` | N/A |\n"

    return table_md + "\n"


def generate_table_for_nested_dict(nested_dict):
    """Generates a Markdown table for a nested dictionary."""
    table_md = "| Key | GUI Name | Type | Default | Description |\n"
    table_md += "|---|---|---|---|---|\n"

    for key, value in nested_dict.items():
        if isinstance(value, dict):
            table_md += f"| `{key}` | {value.get('gui_name', 'N/A')} | `{value.get('type', 'N/A')}` | `{value.get('default', 'N/A')}` | {value.get('description', 'N/A')} |\n"

    return table_md + "\n"


# Generate and save markdown
markdown_output = generate_markdown({**DB, **SETTINGS})

with open("docs/settings.md", "w") as f:
    f.write(markdown_output)

print("Markdown file generated successfully!")
