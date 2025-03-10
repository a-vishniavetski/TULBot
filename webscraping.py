# scraper programy.p.lodz.pl
from bs4 import BeautifulSoup
import json
import re
import requests


def scrape_major(url):
    """
    Scrapes the whole major (field of study)
    :param url: url to the major data
    :type url: str
    :return: all parsed and extracted data
    :rtype: dictionary
    """
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html5lib", from_encoding='utf-8')

    base_major = {'semesters': []}
    # get major data as the HTML
    semester_divs = soup.find_all("div", attrs={'class': 'iform'})

    # extracting base major subjects (from 1 to 4 semester)
    for semester in semester_divs[:4]:
        # TODO nie są brane pod uwagę nie liczbowe semestry a tabeli przdmiotow obieralnych, zrobic to
        # extracting usual semester data
        semester_data = extract_semester_data(semester)
        base_major['semesters'].append(semester_data)

    print(json.dumps(base_major, indent=4))

    # extracting other subjects from specialties
    # if there are no specialties then add subjects to the base major

    # TODO iterate over all specialties if any and add to the separate jsons
    # TODO if there are not specialties add to the major json object
    # TODO save to the json file the extracted data

    return "dupa"


def extract_semester_data(semester: str):
    """
    Extracts data from the usual semester table
    :param: Semester data as a html string
    :return: Dictionary with the semester data
    :rtype: Dictionary
    """
    # semester number
    semester_data = {
        'semester': int(semester.find('h3').text.split()[1]),
        'subjects': []
    }
    # extract array of semester's subjects
    table = semester.find('table')
    if table:
        tbody = table.find('tbody')
        if tbody:
            subjects = tbody.find_all('tr')

            # for every subject extract data related to the subject (name and data from overview file)
            for subject_index, subject in enumerate(subjects):
                subject_name = subject.find('td', attrs={'class': 'w4'}).text
                semester_data['subjects'].append({'subject_name': subject_name})
                raw_onclick_attribute = (subject.find('td', attrs={'class': 'w2 doSrodka'})
                                         .find('a').get('onclick'))
                # extract just the link to the subject overview file
                subject_overview_link = re.search(r"window\.open\('([^']+)'", raw_onclick_attribute)
                if subject_overview_link:
                    # extract data from the overview file
                    subject_overview = parse_subject_overview_file(
                        'https://programy.p.lodz.pl/ectslabel-web/' +
                        subject_overview_link.group(1).replace(' ', '%20')
                    )
                    semester_data['subjects'][subject_index].update(subject_overview)

    return semester_data


def parse_subject_overview_file(link):
    """"
    Parse the subject overview file and return as a dictionary
    Extracts next features:
        - Język prowadzenia zajęć
        - Przedmiotowe efekty uczenia się
        - Kierunkowe efekty uczenia się
        - Szczegółowe treści przedmiotu
    @:param Full link to the overview file
    """
    r = requests.get(link)
    soup = BeautifulSoup(r.content, "html5lib", from_encoding='utf-8')
    table_rows = soup.find('table').find('tbody').find_all('tr')
    subject_overview_data = {}
    for row in table_rows:
        # extract only needed data
        data = row.find_all(recursive=False)
        if data[0].text == 'Język prowadzenia zajęć':
            subject_overview_data['lecture_language'] = data[1].text
        elif data[0].text == 'Przedmiotowe efekty uczenia się':
            subject_overview_data['subject_effects'] = data[1].text
        elif data[0].text == 'Kierunkowe efekty uczenia się':
            subject_overview_data['major_study_effects'] = data[1].text
        elif data[0].text == 'Szczegółowe treści przedmiotu':
            subject_overview_data['subject_content'] = data[1].text

    return subject_overview_data


scrape_major(''
             'https://programy.p.lodz.pl/ectslabel-web/'
             'kierunekSiatkaV4.jsp?l=pl&w=informatyka%20stosowana'
             '&pkId=1654&p=6909&stopien=studia%20pierwszego%20stopnia'
             '&tryb=studia%20stacjonarne&v=4'
             )
