import sys
import xml.etree.ElementTree as ET
from typing import List 
from io import StringIO


def parse_grobid_tei(tei_path):
    ns = {'tei': 'http://www.tei-c.org/ns/1.0', 'xlink': 'http://www.w3.org/1999/xlink'}
    tree = ET.parse(tei_path)
    root = tree.getroot()

    # Title
    title = root.find('.//tei:titleStmt/tei:title', ns)
    title_text = title.text.strip() if title is not None and title.text else ""

    # Authors
    authors = []
    for author in root.findall('.//tei:author', ns):
        surname = author.find('.//tei:surname', ns)
        forename = author.find('.//tei:forename', ns)
        name = " ".join(filter(None, [forename.text if forename is not None else None, surname.text if surname is not None else None]))
        if name:
            authors.append(name)
    authors_text = ", ".join(authors)

    # Abstract
    abstract = root.find('.//tei:abstract', ns)
    abstract_text = " ".join(abstract.itertext()).strip() if abstract is not None else ""

    # Sections
    sections = []
    for div in root.findall('.//tei:text/tei:body/tei:div', ns):
        header = div.find('tei:head', ns)
        section_title = header.text.strip() if header is not None and header.text else ""
        paragraphs = [" ".join(p.itertext()).strip() for p in div.findall('tei:p', ns) if p is not None]
        # Tables in section
        tables = extract_tables(div, ns)
        # Figures in section (with images)
        figures = extract_figures(div, ns)
        # Formulas in section
        formulas = extract_formulas(div, ns)
        sections.append({
            'title': section_title,
            'paragraphs': paragraphs,
            'tables': tables,
            'figures': figures,
            'formulas': formulas
        })

    # References
    references = []
    for bibl in root.findall('.//tei:listBibl/tei:biblStruct', ns):
        ref_title = bibl.find('.//tei:title', ns)
        ref_authors = [a.text for a in bibl.findall('.//tei:author/tei:surname', ns) if a.text]
        ref_str = ""
        if ref_title is not None and ref_title.text:
            ref_str += ref_title.text
        if ref_authors:
            ref_str += " (" + ", ".join(ref_authors) + ")"
        if ref_str:
            references.append(ref_str)

    return {
        'title': title_text,
        'authors': authors_text,
        'abstract': abstract_text,
        'sections': sections,
        'references': references
    }

def extract_tables(parent, ns) -> List[str]:
    tables_md = []
    for table in parent.findall('.//tei:table', ns):
        caption = table.find('tei:head', ns)
        caption_text = caption.text.strip() if caption is not None and caption.text else ""
        rows = []
        for row in table.findall('.//tei:row', ns):
            cells = [cell.text.strip() if cell.text else "" for cell in row.findall('tei:cell', ns)]
            rows.append(cells)
        if rows:
            header = rows[0]
            md = f"\n**Table:** {caption_text}\n\n"
            md += "| " + " | ".join(header) + " |\n"
            md += "|" + "---|" * len(header) + "\n"
            for row in rows[1:]:
                md += "| " + " | ".join(row) + " |\n"
            tables_md.append(md)
    return tables_md

def extract_figures(parent, ns) -> List[str]:
    figures_md = []
    for fig in parent.findall('.//tei:figure', ns):
        desc = fig.find('tei:figDesc', ns)
        caption = desc.text.strip() if desc is not None and desc.text else ""
        graphic = fig.find('tei:graphic', ns)
        img_md = ""
        if graphic is not None:
            # Try both 'url' and '{http://www.w3.org/1999/xlink}href'
            img_url = graphic.get('url') or graphic.get('{http://www.w3.org/1999/xlink}href')
            if img_url:
                img_md = f"![{caption}]({img_url})\n"
        if not img_md:
            img_md = f"**Figure:** {caption}\n"
        figures_md.append(img_md)
    return figures_md

def extract_formulas(parent, ns) -> List[str]:
    formulas_md = []
    for formula in parent.findall('.//tei:formula', ns):
        # Try to extract LaTeX if present, else fallback to MathML or text
        latex = None
        for child in formula:
            if child.tag.endswith('latex'):
                latex = child.text
                break
        if not latex:
            latex = " ".join(formula.itertext()).strip()
        if latex:
            formulas_md.append(f"\n$$\n{latex}\n$$")
    return formulas_md

def to_markdown(parsed) -> str:
    """Converts the parsed dictionary into a single Markdown string."""
    # Use StringIO to build the string in memory
    with StringIO() as f:
        f.write(f"# {parsed['title']}\n\n")
        if parsed['authors']:
            f.write(f"**Authors:** {parsed['authors']}\n\n")
        if parsed['abstract']:
            f.write(f"## Abstract\n{parsed['abstract']}\n\n")
        
        for section in parsed['sections']:
            if section['title']:
                f.write(f"## {section['title']}\n\n")
            for para in section['paragraphs']:
                f.write(para + "\n\n")
            for table_md in section['tables']:
                f.write(table_md + "\n")
            for fig_md in section['figures']:
                f.write(fig_md + "\n")
            for formula_md in section['formulas']:
                f.write(formula_md + "\n")
        
        if parsed['references']:
            f.write("\n## References\n")
            for ref in parsed['references']:
                f.write(f"- {ref}\n")
        
        return f.getvalue()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python grobid_to_markdown.py input.xml output.md")
        sys.exit(1)
    tei_path = sys.argv[1]
    output_path = sys.argv[2]
    parsed = parse_grobid_tei(tei_path)
    to_markdown(parsed, output_path) 