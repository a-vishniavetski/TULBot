# scraper programy.p.lodz.pl
from bs4 import BeautifulSoup
import json
import re
from bs4.element import Tag
import requests
import time
from urllib.parse import urlparse, parse_qs


def get_html_page(url, retries=5, delay=10):
    """
    Access the link with respective delays and retries if failed in order to prevent
    from failing to fetch the page due to being blocked
    :param delay:
    :type delay:
    :param retries:
    :type retries:
    :param url: url to the webpage
    :type url:
    :return: webpage as a html
    :rtype: str
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/123.0.0.0 Safari/537.36"
    }
    # retry retries time the request with respective growing delay
    for attempt in range(retries):
        try:
            print(f"Attempt {attempt + 1}: Fetching {url}...")
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # increase delay time
    print(f'Failed to load the webpage {url}')
    return None


def scrape(url):
    """
    Scrapes the whole major (field of study) Some assumptions: Every semester is a table that as the rows has the
    subjects with their data Sometimes the table is not the semester but the optional subjects table, in that case
    they are added to the previous normal semester :param url: url to the major data :type url: str :return: all
    parsed and extracted data :rtype: dictionary
    """
    major_name = extract_major_name(url)

    webpage = get_html_page(url)
    if webpage is None:
        print(f"Failed to fetch page: {url}")
        return {}

    soup = BeautifulSoup(webpage, "html5lib", from_encoding='utf-8')

    # find specialization form, checks if any choice, store choices in spec_values
    spec_form = soup.find("select", {"name": "sp"})
    spec_values = []
    if spec_form:
        options = spec_form.find_all("option")
        for option in options:
            value = option["value"]
            text = option.text.strip()
            spec_values.append({"value": value, "text": text})

    # scrape major and specializations
    if len(spec_values) == 0:
        scrape_major(url, major_name)
        print("scrapped major without spec")
    else:
        for spec in spec_values:
            spec_url = url + f"&s={spec["value"]}"
            spec_name = major_name + "_" + spec["text"]
            scrape_major(spec_url, spec_name)
            print(f"scrapped {spec["text"]}")


def scrape_major(url, filename):
    webpage = get_html_page(url)
    if webpage is None:
        print(f"Failed to fetch page: {url}")
        return {}

    soup = BeautifulSoup(webpage, "html5lib", from_encoding='utf-8')

    base_major = {'semesters': []}
    # get major data as the HTML
    semester_divs = soup.find_all("div", attrs={'class': 'iform'})

    # extracting base major subjects (from 1st to last semester)
    for semester in semester_divs:
        # if int(semester.find('h3').text.split()[1]) == 5:
        #     break
        # if the provided semester is not full semester but the optional subjects
        if not semester.find("h3"):  # if there are no semesters left
            break
        if 'obieralne' in semester.find("h3").text:
            # parse the optional subjects
            optional_subjects_data = parse_optional_subjects(semester)
            # add optional subjects as the subdata in the respective subject 'Przedmioty obieralne'
            for subject in base_major['semesters'][len(base_major['semesters']) - 1]['subjects']:
                if subject['subject_name'] in semester.find("h3").text:
                    subject['optional_subjects'] = optional_subjects_data
                    break
        else:
            # extracting usual semester table data
            semester_data = extract_semester_data(semester)
            base_major['semesters'].append(semester_data)

    save_to_json(filename, base_major)


def save_to_json(filename, data):
    with open(f'{filename}.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def extract_semester_data(semester: str):
    """
    Extracts data from the usual semester table
    :param: Semester table data as a html string
    :return: Dictionary with the semester data
    :rtype: Dictionary
    """
    # semester number
    semester_data = {}
    if 'obieralne' not in semester.find("h3").text:  # if it is normal semester data
        semester_data['semester'] = int(semester.find('h3').text.split()[1])
    semester_data['subjects'] = []
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
                try:
                    raw_onclick_attribute = (subject.find('td', attrs={'class': 'w2'})
                                             .find('a').get('onclick'))
                except AttributeError:
                    # if there is no overview file or the subject is about optional subjects then
                    # check if it is the optional subjects subject
                    if 'obieralne' in subject.find('td', attrs={'class': 'w2 doSrodka'}).text:
                        semester_data['subjects'][subject_index]['subject_overview'] = 'Przedmioty obieralne'
                    else:
                        semester_data['subjects'][subject_index]['subject_overview'] = 'No overview for this subject'
                    continue

                # extract just the link to the subject overview file
                subject_overview_link = re.search(r"window\.open\('([^']+)'", raw_onclick_attribute)
                if subject_overview_link:
                    # extract data from the overview file
                    subject_overview = parse_subject_overview_file(
                        'https://programy.p.lodz.pl/ectslabel-web/' +
                        subject_overview_link.group(1).replace(' ', '%20')
                    )
                    semester_data['subjects'][subject_index]['subject_overview'] = subject_overview

    return semester_data


def parse_optional_subjects(optional_subjects_table):
    return extract_semester_data(optional_subjects_table)


def parse_subject_overview_file(link):
    """"
    Parse the subject overview file and return as a dictionary
    Extracts next features:
        - Kod przedmiotu
        - Język prowadzenia zajęć
        - Przedmiotowe efekty uczenia się
        - Kierunkowe efekty uczenia się
        - Szczegółowe treści przedmiotu
        - Bilans godzin
        - Metody weryfikacji przedmiotowych efektów uczenia się
        - Wymagania wstępne
    @:param Full link to the overview file
    """
    webpage = get_html_page(link)
    soup = BeautifulSoup(webpage, "html5lib", from_encoding='utf-8')
    table_rows = soup.find('table').find('tbody').find_all('tr')
    subject_overview_data = {}
    for row in table_rows:
        # extract only needed data
        data = row.find_all(recursive=False)
        if data[0].text == 'Kod przedmiotu':
            subject_overview_data['subject_id'] = data[1].text
        elif data[0].text == 'Język prowadzenia zajęć':
            subject_overview_data['lecture_language'] = data[1].text
        elif data[0].text == 'Przedmiotowe efekty uczenia się':
            subject_overview_data['subject_effects'] = data[1].text
        elif data[0].text == 'Kierunkowe efekty uczenia się':
            subject_overview_data['major_study_effects'] = data[1].text
        elif data[0].text == 'Szczegółowe treści przedmiotu':
            subject_overview_data['subject_content'] = data[1].text
        elif data[0].text == 'Metody weryfikacji przedmiotowych efektów uczenia się':
            subject_overview_data['subject_effects_verification'] = data[1].text
        elif data[0].text == 'Wymagania wstępne':
            subject_overview_data['prerequisites'] = data[1].text
        elif 'Bilans godzin' in data[0].text:
            subject_overview_data['time_distribution'] = {}
            all_lesson_types = data[1].find('table').find_all('tbody')
            all_lesson_types = [lesson_type.find('tr') for lesson_type in all_lesson_types]
            for lesson_type in all_lesson_types:
                if 'SUMA' in lesson_type.find('th'):
                    subject_overview_data['time_distribution']['total'] = int(lesson_type.find('td').text)
                    continue
                lesson_type_name = lesson_type.find('th').text.replace(' ', '_')
                subject_overview_data['time_distribution'][lesson_type_name] = int(lesson_type.find('td').text)

                # if lesson_type.find('th') == 'Wykład':
                #     subject_overview_data['time_distribution']['lecture'] = int(lesson_type.find('td').text)
                # elif lesson_type.find('th') == 'Ćwiczenia':
                #     subject_overview_data['time_distribution']['tutorial'] = int(lesson_type.find('td').text)
                # elif lesson_type.find('th') == 'Laboratorium':
                #     subject_overview_data['time_distribution']['laboratory'] = int(lesson_type.find('td').text)
                # elif lesson_type.find('th') == 'E-learning':
                #     subject_overview_data['time_distribution']['e_learning'] = int(lesson_type.find('td').text)
                # elif lesson_type.find('th') == 'Realizacja zadań laboratoryjnych':
                #     subject_overview_data['time_distribution']['laboratory_tasks'] = int(lesson_type.find('td').text)
                # elif lesson_type.find('th') == 'Przygotowanie prac projektowych':
                #     subject_overview_data['time_distribution']['project_tasks_preparation'] = int(
                #         lesson_type.find('td').text)
                # elif lesson_type.find('th') == 'Projekt':
                #     subject_overview_data['time_distribution']['project'] = int(lesson_type.find('td').text)
                # elif lesson_type.find('th') == 'Ćwiczenia z wykorzystaniem komputera':
                #     subject_overview_data['time_distribution']['computer_tasks'] = int(lesson_type.find('td').text)
                # elif lesson_type.find('th') == 'Seminarium':
                #     subject_overview_data['time_distribution']['seminar'] = int(lesson_type.find('td').text)
                # elif 'Nauka własna' in lesson_type.find('th')\
                #         or 'Praca własna' in lesson_type.find('th'):
                #     subject_overview_data['time_distribution']['self_teaching'] = int(lesson_type.find('td').text)
                # elif lesson_type.find('th') == 'Przygotowanie prezentacji':
                #     subject_overview_data['time_distribution']['presentation_preparation'] = int(
                #         lesson_type.find('td').text)
                # elif lesson_type.find('th') == 'Przygotowanie do egzaminu':
                #     subject_overview_data['time_distribution']['exam_preparation'] = int(lesson_type.find('td').text)
                # elif lesson_type.find('th') == 'Przygotowanie do zaliczenia ćwiczeń'\
                #         or lesson_type.find('th') == 'Przygotowanie do ćwiczeń':
                #     subject_overview_data['time_distribution']['tutorial_preparation'] = int(
                #         lesson_type.find('td').text)
                # elif lesson_type.find('th') == 'Przygotowanie do kolokwiów':
                #     subject_overview_data['time_distribution']['final_test_preparation'] = int(
                #         lesson_type.find('td').text)
                # elif lesson_type.find('th') == 'Przygotowanie do testu pisemnego':
                #     subject_overview_data['time_distribution']['written_test_preparation'] = int(
                #         lesson_type.find('td').text)
                # elif lesson_type.find('th') == 'Przygotowanie do zajęć laboratoryjnych':
                #     subject_overview_data['time_distribution']['laboratory_preparation'] = int(
                #         lesson_type.find('td').text)
                # elif lesson_type.find('th') == 'Przygotowanie sprawozdań z ćwiczeń laboratoryjnych':
                #     subject_overview_data['time_distribution']['laboratory_report_preparation'] = int(
                #         lesson_type.find('td').text)
                # elif lesson_type.find('th') == 'Przygotowanie do dyskusji':
                #     subject_overview_data['time_distribution']['discussion_preparation'] = int(
                #         lesson_type.find('td').text)
                # elif 'literatury' in lesson_type.find('th') \
                #         or 'bibliotece' in lesson_type.find('th')\
                #         or 'biblioteka' in lesson_type.find('th')\
                #         or 'literatura' in lesson_type.find('th'):
                #     subject_overview_data['time_distribution']['literature_reading'] = int(lesson_type.find('td').text)
                # elif lesson_type.find('th') == 'Studiowanie materiałów źródłowych' \
                #         or lesson_type.find('th') == 'Praca z materiałami źródłowymi':
                #     subject_overview_data['time_distribution']['source_material_reading'] = int(
                #         lesson_type.find('td').text)
                # elif 'SUMA' in lesson_type.find('th'):
                #     subject_overview_data['time_distribution']['total'] = int(lesson_type.find('td').text)

    return subject_overview_data


def extract_major_name(url):
    """
    Extracts the major name from a given URL.
    Assumes the major name is stored in the 'w' parameter of the URL.
    
    :param url: URL to the major data (e.g., from programy.p.lodz.pl)
    :type url: str
    :return: Name of the major (field of study)
    :rtype: str
    """
    try:
        # Parsowanie URL i ekstrakcja parametrów
        parsed_url = urlparse(url)
        params = parse_qs(parsed_url.query)
        
        # Wyciągnięcie wartości parametru 'w'
        if 'w' in params:
            major_name = params['w'][0]  # Pierwsza wartość parametru 'w'
            # Dekodowanie URL (np. zamiana %20 na spacje)
            major_name = major_name.replace('%20', '_')
            return major_name
        else:
            return "Nazwa kierunku nie znaleziona w URL"
    
    except Exception as e:
        print(f"Błąd podczas parsowania URL: {e}")
        return "Błąd parsowania URL"


informatyka_stosowana = 'https://programy.p.lodz.pl/ectslabel-web/kierunekSiatkaV4.jsp?l=pl&w=informatyka%20stosowana&pkId=1654&p=6909&stopien=studia%20pierwszego%20stopnia&tryb=studia%20stacjonarne&v=4'

analityka_chemiczna = "https://programy.p.lodz.pl/ectslabel-web/kierunekSiatkaV4.jsp?l=pl&w=analityka%20chemiczna&pkId=1687&p=6947&stopien=studia%20pierwszego%20stopnia&tryb=studia%20stacjonarne&v=4"


scrape(informatyka_stosowana)
