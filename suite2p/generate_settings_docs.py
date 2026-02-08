from suite2p.parameters import DB, SETTINGS

def generate_markdown(settings, heading=True):
    """Generates Markdown with optimized section creation for nested dictionaries."""
    md_content = ""
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
        if heading:
            md_content += "### general settings\n\n"
        md_content += generate_table(flat_entries)

    # Function to process nested dictionaries
    def process_nested_dict(nested_dict, level=3):
        """Recursively process nested dictionaries into Markdown sections."""
        md = ""

        for key, value in nested_dict.items():
            section_title = key.replace("_", " ") #.capitalize()

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

with open("docs/parameters.md", "w") as f:
    md_content = "# Db and Settings for Suite2p\n\n"
    md_content += ("Suite2p can be run with different configurations using the db and settings dictionaries. "
                   "The db dictionary contains recording specific parameters, and the settings dictionary contains pipeline parameters. "
                   "Here is a summary of all the parameters that the pipeline takes and their default values.\n\n")
    f.write(md_content)
    f.write("## db.npy\n\n")
    markdown_output = generate_markdown(DB, heading=False)
    f.write(markdown_output)
    f.write("## settings.npy\n\n")
    markdown_output = generate_markdown(SETTINGS)
    f.write(markdown_output)
    
print("Markdown file generated successfully!")
