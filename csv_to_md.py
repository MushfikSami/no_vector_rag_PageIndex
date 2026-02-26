import pandas as pd
import os

def convert_csv_to_markdown(csv_path, output_md_path):
    print(f"Reading data from {csv_path}...")
    
    # Load the CSV, handling any missing values
    df = pd.read_excel(csv_path)
    df = df.fillna("")

    markdown_content = ""
    
    # Track the current hierarchy to avoid repeating headings
    current_cat = ""
    current_subcat = ""
    current_service = ""

    for index, row in df.iterrows():
        cat = str(row.get('Category', '')).strip()
        subcat = str(row.get('Sub-Category', '')).strip()
        service = str(row.get('Service', '')).strip()
        topic = str(row.get('Topic', '')).strip()
        text = str(row.get('Text', '')).strip()
        keywords = str(row.get('Text Keywords', '')).strip()
        
        # Level 1: Category
        if cat and cat != current_cat:
            markdown_content += f"\n# {cat}\n"
            current_cat = cat
            current_subcat = ""  # Reset lower levels
            current_service = ""
        
        # Level 2: Sub-Category
        if subcat and subcat != current_subcat:
            markdown_content += f"\n## {subcat}\n"
            current_subcat = subcat
            current_service = ""
            
        # Level 3: Service
        if service and service != current_service:
            markdown_content += f"\n### {service}\n"
            current_service = service
            
        # Level 4: Topic
        if topic:
            markdown_content += f"\n#### {topic}\n\n"
            
        # Add the metadata and core text
        if keywords:
            markdown_content += f"**Keywords:** {keywords}\n\n"
            
        if text:
            # Replace literal literal \n in the CSV with actual markdown line breaks
            formatted_text = text.replace("\\n", "\n")
            markdown_content += f"{formatted_text}\n\n"
            
        # Add a visual separator between entries
        markdown_content += "---\n"

    # Save with UTF-8 encoding to preserve Bengali characters
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
        
    print(f"✅ Successfully converted to Markdown! Saved as: {output_md_path}")

if __name__ == "__main__":
    # Your specific uploaded file
    INPUT_CSV = "Copy of unpublished_version_03_8 Januay 2026.xlsx"
    OUTPUT_MD = "bangladesh_services_data.md"
    
    convert_csv_to_markdown(INPUT_CSV, OUTPUT_MD)