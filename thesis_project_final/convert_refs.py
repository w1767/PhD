#!/usr/bin/env python3
"""
Script to convert manual LaTeX references from longtable format to BibTeX format.
"""

import re
import unicodedata

def clean_latex(text):
    """Remove LaTeX formatting and clean up text."""
    # Remove \emph{...}
    text = re.sub(r'\\emph\{([^}]*)\}', r'\1', text)
    # Remove \textbf{...}
    text = re.sub(r'\\textbf\{([^}]*)\}', r'\1', text)
    # Remove \url{...} but keep the URL
    text = re.sub(r'\\url\{([^}]*)\}', r'\1', text)
    # Replace -- with -
    text = text.replace('--', '-')
    # Replace \& with &
    text = text.replace('\\&', '&')
    # Remove {[} and {]}
    text = text.replace('{[}', '[').replace('{]}', ']')
    # Remove \textasciitilde
    text = text.replace('\\textasciitilde', '~')
    # Remove \texorpdfstring
    text = re.sub(r'\\texorpdfstring\{[^}]*\}\{([^}]*)\}', r'\1', text)
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

def generate_cite_key(authors, year, title):
    """Generate a citation key from authors, year, and title."""
    # Get first author's last name
    if authors:
        first_author = authors.split(',')[0].split(' and ')[0].strip()
        # Try to get last name (last word before any comma or period)
        name_parts = first_author.replace('.', '').split()
        if name_parts:
            last_name = name_parts[-1] if len(name_parts) > 1 else name_parts[0]
            # Remove non-alphanumeric
            last_name = re.sub(r'[^a-zA-Z]', '', last_name).lower()
        else:
            last_name = 'unknown'
    else:
        last_name = 'unknown'
    
    # Get first significant word from title
    if title:
        title_words = title.split()
        skip_words = {'a', 'an', 'the', 'on', 'in', 'of', 'for', 'and', 'to', 'with'}
        title_word = ''
        for word in title_words:
            clean_word = re.sub(r'[^a-zA-Z]', '', word).lower()
            if clean_word and clean_word not in skip_words:
                title_word = clean_word
                break
        if not title_word and title_words:
            title_word = re.sub(r'[^a-zA-Z]', '', title_words[0]).lower()
    else:
        title_word = 'untitled'
    
    # Extract year
    year_str = str(year) if year else 'nd'
    
    return f"{last_name}{year_str}{title_word}"

def parse_reference(ref_text, ref_num):
    """Parse a single reference and return a BibTeX entry."""
    ref_text = clean_latex(ref_text)
    
    # Initialize fields
    entry_type = 'misc'
    fields = {}
    
    # Try to extract year (4 digits, typically 19xx or 20xx)
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', ref_text)
    year = year_match.group(1) if year_match else None
    fields['year'] = year if year else 'n.d.'
    
    # Check for online reference
    if '[Online]' in ref_text or 'Available:' in ref_text:
        entry_type = 'online'
        # Extract URL
        url_match = re.search(r'Available:\s*(https?://[^\s\[\]]+)', ref_text)
        if url_match:
            fields['url'] = url_match.group(1).rstrip('.')
        # Extract access date
        access_match = re.search(r'\[Accessed\s+([^\]]+)\]', ref_text)
        if access_match:
            fields['urldate'] = access_match.group(1)
    
    # Check for "in" indicating conference/proceedings
    elif ' in ' in ref_text.lower() and ('proceedings' in ref_text.lower() or 
                                          'conference' in ref_text.lower() or
                                          'symposium' in ref_text.lower()):
        entry_type = 'inproceedings'
        # Try to extract booktitle
        in_match = re.search(r'\bin\s+(.+?),\s*(?:\d{4}|[A-Z][a-z]+\s+\d{4})', ref_text, re.IGNORECASE)
        if in_match:
            fields['booktitle'] = in_match.group(1).strip()
    
    # Check for book (has "ed." for edition, or publisher pattern without journal vol/no)
    elif re.search(r'\d+\s*(?:st|nd|rd|th)\s+ed\.?', ref_text, re.IGNORECASE) or \
         (': ' in ref_text and 'vol.' not in ref_text.lower() and 'pp.' not in ref_text.lower()):
        entry_type = 'book'
        # Try to extract publisher
        pub_match = re.search(r':\s*([^,]+(?:Press|Publishing|Wiley|Springer|Elsevier|McGraw|Sons)[^,]*)', ref_text)
        if pub_match:
            fields['publisher'] = pub_match.group(1).strip()
        # Edition
        ed_match = re.search(r'(\d+)\s*(?:st|nd|rd|th)\s+ed\.?', ref_text, re.IGNORECASE)
        if ed_match:
            fields['edition'] = ed_match.group(1)
    
    # Check for article (has vol., no., pp.)
    elif 'vol.' in ref_text.lower() or 'pp.' in ref_text.lower():
        entry_type = 'article'
        # Extract volume
        vol_match = re.search(r'vol\.\s*(\d+)', ref_text, re.IGNORECASE)
        if vol_match:
            fields['volume'] = vol_match.group(1)
        # Extract number
        no_match = re.search(r'no\.\s*([\d-]+)', ref_text, re.IGNORECASE)
        if no_match:
            fields['number'] = no_match.group(1)
        # Extract pages
        pp_match = re.search(r'pp\.\s*([\d\s-]+)', ref_text, re.IGNORECASE)
        if pp_match:
            fields['pages'] = pp_match.group(1).strip().replace(' ', '')
    
    # Try to extract title (text in ``...'' or "...")
    title_match = re.search(r'``([^\']+)\'\'', ref_text)
    if not title_match:
        title_match = re.search(r'"([^"]+)"', ref_text)
    if title_match:
        fields['title'] = title_match.group(1).strip().rstrip(',')
    else:
        # For books, title might be before the first comma after author
        # This is a fallback
        parts = ref_text.split(',')
        if len(parts) > 1:
            potential_title = parts[1].strip()
            if not any(x in potential_title.lower() for x in ['vol.', 'pp.', 'ed.']):
                fields['title'] = potential_title
    
    # Try to extract journal name (italicized text that's not the title)
    # Journal usually comes after the title
    if entry_type == 'article':
        # Look for text after title quotes and before vol.
        after_title = ref_text
        if '\"' in ref_text or "''" in ref_text:
            title_end = max(ref_text.rfind("''"), ref_text.rfind('"'))
            if title_end > 0:
                after_title = ref_text[title_end+2:]
        
        # Journal is usually between commas before vol.
        journal_match = re.search(r',\s*([A-Z][^,]+?),\s*vol\.', after_title, re.IGNORECASE)
        if journal_match:
            fields['journal'] = journal_match.group(1).strip()
    
    # Extract author (everything before the title or first comma with title pattern)
    # Authors typically end before the title (which is in quotes or before a comma followed by title)
    author_text = ref_text
    if 'title' in fields:
        title_pos = ref_text.find(fields['title'])
        if title_pos > 0:
            author_text = ref_text[:title_pos]
    
    # Clean up author text
    author_text = author_text.strip().rstrip(',').strip()
    # Remove trailing punctuation
    author_text = re.sub(r'[,\s``"]+$', '', author_text)
    
    if author_text:
        fields['author'] = author_text
    
    # Generate citation key
    cite_key = generate_cite_key(
        fields.get('author', ''),
        fields.get('year', ''),
        fields.get('title', '')
    )
    # Add ref number to ensure uniqueness
    cite_key = f"{cite_key}_{ref_num}"
    
    # Build BibTeX entry
    lines = [f"@{entry_type}{{{cite_key},"]
    for key, value in fields.items():
        if value:
            # Escape special characters
            value = str(value).replace('{', '\\{').replace('}', '\\}')
            value = value.replace('#', '\\#').replace('_', '\\_')
            # But we want braces around the value
            value = value.replace('\\{', '{').replace('\\}', '}')
            lines.append(f"  {key} = {{{value}}},")
    lines.append("}")
    
    return '\n'.join(lines), cite_key

def extract_references(tex_content):
    """Extract references from the longtable in the LaTeX content."""
    # Find the references section
    start_marker = r'\\begin{longtable}'
    end_marker = r'\\end{longtable}'
    
    # Find the references longtable (the one after \chapter{References})
    ref_chapter = tex_content.find('\\chapter{\\texorpdfstring{\\textbf{References}}')
    if ref_chapter == -1:
        ref_chapter = tex_content.find('\\chapter{References}')
    
    if ref_chapter == -1:
        print("Could not find References chapter")
        return []
    
    # Find longtable after references chapter
    table_start = tex_content.find('\\begin{longtable}', ref_chapter)
    table_end = tex_content.find('\\end{longtable}', table_start)
    
    if table_start == -1 or table_end == -1:
        print("Could not find references longtable")
        return []
    
    table_content = tex_content[table_start:table_end]
    
    # Parse individual references
    # References are in format: {[}N{]} & reference text \\
    references = []
    
    # Split by \\ to get rows
    rows = table_content.split('\\\\')
    
    current_ref_num = None
    current_ref_text = ""
    
    for row in rows:
        # Check if this row starts a new reference
        ref_match = re.search(r'\{?\[?\}?(\d+)\{?\]?\}?\s*&\s*(.+)', row, re.DOTALL)
        if ref_match:
            # Save previous reference if exists
            if current_ref_num is not None and current_ref_text.strip():
                references.append((current_ref_num, current_ref_text.strip()))
            
            current_ref_num = int(ref_match.group(1))
            current_ref_text = ref_match.group(2)
        elif current_ref_num is not None:
            # This might be continuation of previous reference
            # But usually references are on single lines in this format
            pass
    
    # Don't forget the last reference
    if current_ref_num is not None and current_ref_text.strip():
        references.append((current_ref_num, current_ref_text.strip()))
    
    return references

def main():
    # Read the main.tex file
    with open('main.tex', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract references
    references = extract_references(content)
    print(f"Found {len(references)} references")
    
    # Convert to BibTeX
    bib_entries = []
    cite_keys = []
    
    for ref_num, ref_text in references:
        try:
            bib_entry, cite_key = parse_reference(ref_text, ref_num)
            bib_entries.append(bib_entry)
            cite_keys.append((ref_num, cite_key))
            print(f"[{ref_num}] -> {cite_key}")
        except Exception as e:
            print(f"Error parsing reference {ref_num}: {e}")
            # Create a minimal entry
            bib_entry = f"@misc{{ref{ref_num},\n  note = {{{clean_latex(ref_text)}}},\n}}"
            bib_entries.append(bib_entry)
            cite_keys.append((ref_num, f"ref{ref_num}"))
    
    # Write BibTeX file
    with open('references.bib', 'w', encoding='utf-8') as f:
        f.write("% Bibliography generated from manual references\n")
        f.write("% Generated by convert_refs.py\n\n")
        f.write('\n\n'.join(bib_entries))
    
    print(f"\nGenerated references.bib with {len(bib_entries)} entries")
    
    # Write a mapping file for reference
    with open('cite_key_mapping.txt', 'w', encoding='utf-8') as f:
        f.write("Reference Number -> Citation Key Mapping\n")
        f.write("=" * 50 + "\n\n")
        for ref_num, cite_key in cite_keys:
            f.write(f"[{ref_num}] -> \\cite{{{cite_key}}}\n")
    
    print("Generated cite_key_mapping.txt with citation key mappings")

if __name__ == '__main__':
    main()

