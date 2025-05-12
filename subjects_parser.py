import json


def transform_json(input_json):
    subjects_dict = {}
    major = input_json['major']
    spec = input_json['specialization']
    mode = input_json['mode']
    lang = ['lang']
    for semester_data in input_json['semesters']:
        semester_number = semester_data['semester']

        for subject in semester_data.get('subjects', []):
            # Extract subject details
            subject_details = subject.get('subject_overview', {})

            # Create subject entry if not exists
            subject_name = subject.get('subject_name', '')
            subject_id = subject_details.get('subject_id', '')

            if subject_id not in subjects_dict:
                subjects_dict[subject_id] = {
                    "Nazwa przedmiotu": subject_name,
                    "Kierunki na których się pojawia": [major],
                    "Specjalizacji na których się pojawia": [spec],
                    "Język prowadzenia": lang,
                    "Semestr": semester_number,
                    "Tryb kierunku": mode
                }
    output_json = {
        "subjects": subjects_dict
    }

    return output_json
