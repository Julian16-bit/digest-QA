"""
Web scraper for extracting text content from Canada.ca Employment Insurance pages.
Includes chapter extraction and integration functionality.
"""

from bs4 import BeautifulSoup
import requests
import re
import json
import unicodedata


def extract_text_from_pages(start_url, max_pages=40):
    """
    Extract text from multiple pages following pagination.

    Args:
        start_url: Initial URL to start scraping
        max_pages: Maximum number of pages to scrape

    Returns:
        list: List of page text content
    """
    i = 0
    page_list = []

    while i < max_pages:
        response = requests.get(start_url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser", from_encoding=response.encoding)

            page_text = soup.find("div", class_="mwsgeneric-base-html parbase section")

            if page_text:
                for panel_body in page_text.find_all('div', class_='panel-body'):
                    panel_body.extract()

            page_list.append(page_text.text)

            next_button = soup.find("a", href=True, rel="next")
            if next_button:
                start_url = "https://www.canada.ca" + next_button['href']
                i += 1
            else:
                break
        else:
            print("Failed to fetch", start_url)
            break

    return page_list


def split_text(input_text):
    """
    Split text into structured sections based on section numbers.

    Args:
        input_text: Raw text content from page

    Returns:
        str: JSON formatted sections
    """
    sections = re.split(r'\n(\d+\.\d+\.\d+\.?\d?)\s', input_text.strip())

    formatted_sections = []

    for i in range(1, len(sections), 2):
        section_number = sections[i]
        section_content = sections[i + 1].split('\n', 1)
        section_title = section_content[0].strip()
        section_text = section_content[1].strip() if len(section_content) > 1 else ''

        formatted_sections.append({
            "section_number": section_number,
            "section_title": section_title,
            "section_text": section_text
        })

    return json.dumps(formatted_sections, indent=2, ensure_ascii=False)


def normalize_unicode(text):
    """
    Normalize Unicode characters to ASCII representation.

    Args:
        text: Text with Unicode characters

    Returns:
        str: ASCII normalized text
    """
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')


def scrape_chapter_list(url='https://www.canada.ca/fr/emploi-developpement-social/programmes/assurance-emploi/ae-liste/rapports/guide.html'):
    """
    Scrape the list of chapters from the EI Digest main page.

    Args:
        url: URL of the EI Digest main page

    Returns:
        list: List of chapter titles
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    list_parent = soup.find('div', class_='panel-body')
    list_items = list_parent.find_all('li')

    chapter_list = []
    for item in list_items:
        chapter_list.append(item.get_text(strip=True))

    return chapter_list


def get_chapter_from_section_number(section_number, chapter_list):
    """
    Get the chapter title for a given section number.

    Args:
        section_number: Section number (e.g., "10.5.2")
        chapter_list: List of chapter titles

    Returns:
        str: Chapter title
    """
    first_number = int(section_number.split('.')[0])
    return chapter_list[first_number - 1]


def add_chapters_to_sections(sections_data, chapter_list):
    """
    Add chapter information to sections list.

    Args:
        sections_data: List of section dictionaries
        chapter_list: List of chapter titles

    Returns:
        list: Sections with chapter information added
    """
    for section in sections_data:
        first_number = int(section['section_number'].split('.')[0])
        section['section_chapter'] = chapter_list[first_number - 1]

    return sections_data


def extract_meta_from_pages(start_url, max_pages=50):
    """
    Extract metadata (descriptions and keywords) from multiple pages.

    Args:
        start_url: Initial URL to start scraping
        max_pages: Maximum number of pages to scrape

    Returns:
        tuple: (descriptions list, keywords list)
    """
    descriptions = []
    keywords = []
    i = 0

    while i < max_pages:
        response = requests.get(start_url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            title_parent = soup.find('div', class_='mwstitle section')
            title = title_parent.find("h1").text
            description_meta = soup.find("meta", attrs={"name": "description"})
            keyword_meta = soup.find("meta", attrs={"name": "keywords"})

            if 'Section' in title:
                if description_meta:
                    description_content = description_meta['content']
                    descriptions.append(description_content)

                if keyword_meta:
                    keyword_content = keyword_meta['content']
                    keywords.append(keyword_content)

            next_button = soup.find("a", href=True, rel="next")

            if next_button:
                start_url = "https://www.canada.ca" + next_button['href']
                i += 1
            else:
                break
        else:
            print("Failed to fetch", start_url)
            break

    return descriptions, keywords


def save_scraped_text(start_url, output_file, max_pages=40, include_chapters=True, chapter_url=None):
    """
    Scrape pages and save formatted sections to file with optional chapter integration.

    Args:
        start_url: Initial URL to start scraping
        output_file: Output file path (JSON format)
        max_pages: Maximum number of pages to scrape
        include_chapters: Whether to include chapter information
        chapter_url: URL to scrape chapters from (if different from default)
    """
    full_text = extract_text_from_pages(start_url, max_pages)

    all_sections = []
    for page_text in full_text:
        sections_json = split_text(page_text)
        sections = json.loads(sections_json)
        all_sections.extend(sections)

    if include_chapters:
        chapter_list = scrape_chapter_list(chapter_url) if chapter_url else scrape_chapter_list()
        all_sections = add_chapters_to_sections(all_sections, chapter_list)

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(all_sections, file, indent=4, ensure_ascii=False)


def save_meta_to_csv(start_url, output_csv, max_pages=50):
    """
    Extract metadata and save to CSV file.

    Args:
        start_url: Initial URL to start scraping
        output_csv: Output CSV file path
        max_pages: Maximum number of pages to scrape
    """
    import csv

    descriptions, keywords = extract_meta_from_pages(start_url, max_pages)

    with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        if csvfile.tell() == 0:
            writer.writerow(['Description', 'Keywords'])

        for desc, key in zip(descriptions, keywords):
            writer.writerow([desc, key])


if __name__ == "__main__":
    # Example usage - French version with chapters
    french_start_url = "https://www.canada.ca/en/employment-social-development/programs/ei/ei-list/reports/digest/chapter-1/authority.html#a1_1_0"
    save_scraped_text(french_start_url, 'output_french_full.json', include_chapters=True)

    # Example usage - English version metadata
    # english_start_url = 'https://www.canada.ca/en/employment-social-development/programs/ei/ei-list/reports/digest/chapter-25/authority.html'
    # save_meta_to_csv(english_start_url, 'meta_data.csv')
