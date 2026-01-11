"""
Script to add chapter information to scraped sections from the EI Digest.
"""

from bs4 import BeautifulSoup
import requests
import json


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


def add_chapters_to_sections(input_file, output_file, chapter_list):
    """
    Read sections from file and add chapter information.

    Args:
        input_file: Path to input file with scraped sections
        output_file: Path to output JSON file
        chapter_list: List of chapter titles
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    section = []
    section_title = []
    section_text = []
    section_chapter = []

    for line in lines:
        line_parts = line.split(':', 1)
        if 'section_number' in line_parts[0]:
            section_number = line_parts[1].strip().replace(',', '')
            section.append(section_number)
            first_number = int(section_number.split('.')[0])
            chapter = chapter_list[first_number - 1]
            section_chapter.append(chapter)
        elif 'section_title' in line_parts[0]:
            section_title.append(line_parts[1].strip().replace(',', ''))
        elif 'section_text' in line_parts[0]:
            section_text.append(line_parts[1].strip())
        else:
            pass

    sections = []
    for num, title, text, chapter in zip(section, section_title, section_text, section_chapter):
        section_obj = {
            'section_number': num,
            'section_title': title,
            'section_chapter': chapter,
            'section_text': text
        }
        sections.append(section_obj)

    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(sections, json_file, indent=4, ensure_ascii=False)


def add_chapters_to_json(input_json, output_json, chapter_list):
    """
    Add chapter information to existing JSON data.

    Args:
        input_json: Path to input JSON file
        output_json: Path to output JSON file
        chapter_list: List of chapter titles
    """
    with open(input_json, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for section in data:
        first_number = int(section['section_number'].split('.')[0])
        section_chapter = chapter_list[first_number - 1]
        section['section_chapter'] = section_chapter

    with open(output_json, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # Scrape chapter list from main page
    chapter_list = scrape_chapter_list()

    print(f"Found {len(chapter_list)} chapters")
    print("\nExample chapters:")
    print(f"Chapter 10: {chapter_list[9]}")
    print(f"Chapter 5: {chapter_list[4]}")

    # Example: Add chapters to sections from text file
    # add_chapters_to_sections('output_french_full.txt', 'formatted_sections_french_full.json', chapter_list)

    # Example: Add chapters to existing JSON
    # add_chapters_to_json('input.json', 'output_with_chapters.json', chapter_list)
