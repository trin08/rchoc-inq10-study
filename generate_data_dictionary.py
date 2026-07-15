"""
Generates the REDCap Data Dictionary for the RCHOC INQ-10 study --
every field except the INQ-10 itself, which lives in its own
instrument (see generate_inq10_instrument.py) so it can carry its
own pre-disclosure text and matrix layout.

Field names here match config.COLUMNS exactly, so a REDCap export
built from this dictionary needs zero edits to config.py before
analyze_redcap.py can read it.

    python3 generate_data_dictionary.py --output rchoc_data_dictionary.csv
"""

import argparse
import csv

HEADERS = [
    "Variable / Field Name", "Form Name", "Section Header", "Field Type",
    "Field Label", "Choices, Calculations, OR Slider Labels", "Field Note",
    "Text Validation Type OR Show Slider Number", "Text Validation Min",
    "Text Validation Max", "Identifier?", "Branching Logic (Show field only if...)",
    "Required Field?", "Custom Alignment", "Question Number (surveys only)",
    "Matrix Group Name", "Matrix Ranking?", "Field Annotation",
]


def build_fields():
    fields = []

    # --- demographics -------------------------------------------------
    # subject_id is the coded, non-identifying Subject ID (e.g. Sub-0001)
    # assigned per protocol -- the MRN/name link lives only in the
    # separate Master Linking Log and never enters this database, so
    # this field is correctly NOT flagged as an identifier.
    fields += [
        {
            "Variable / Field Name": "subject_id",
            "Form Name": "demographics",
            "Field Type": "text",
            "Field Label": "Subject ID (coded, non-sequential -- see Master Linking Log)",
            "Identifier?": "n",
            "Required Field?": "y",
        },
        {
            "Variable / Field Name": "age",
            "Form Name": "demographics",
            "Field Type": "text",
            "Field Label": "Age at ED presentation (years)",
            "Field Note": "Computed from DOB during chart abstraction, outside REDCap -- "
                           "DOB itself is a direct identifier and does not enter this database.",
            "Text Validation Type OR Show Slider Number": "integer",
            "Text Validation Min": "12",
            "Text Validation Max": "21",
            "Identifier?": "n",
            "Required Field?": "y",
        },
        {
            "Variable / Field Name": "biological_sex",
            "Form Name": "demographics",
            "Field Type": "radio",
            "Field Label": "Biological sex",
            "Choices, Calculations, OR Slider Labels": "0, Female | 1, Male",
            "Identifier?": "n",
            "Required Field?": "y",
        },
        {
            "Variable / Field Name": "gender",
            "Form Name": "demographics",
            "Field Type": "radio",
            "Field Label": "Gender",
            "Choices, Calculations, OR Slider Labels": "male, Male | female, Female | other, Other / not listed",
            "Identifier?": "n",
        },
        {
            "Variable / Field Name": "race_ethnicity",
            "Form Name": "demographics",
            "Field Type": "text",
            "Field Label": "Race/ethnicity (self-reported, as documented in EHR)",
            "Field Note": "Descriptive use only -- excluded from the confirmatory model per protocol.",
            "Identifier?": "n",
        },
        {
            "Variable / Field Name": "insurance_type",
            "Form Name": "demographics",
            "Field Type": "radio",
            "Field Label": "Insurance type (SES proxy)",
            "Choices, Calculations, OR Slider Labels": "private, Private | medicaid, Medicaid | uninsured, Uninsured",
            "Identifier?": "n",
        },
        {
            "Variable / Field Name": "zip_code",
            "Form Name": "demographics",
            "Field Type": "text",
            "Field Label": "ZIP code (SES proxy)",
            "Text Validation Type OR Show Slider Number": "zipcode",
            "Field Note": "SES proxy for Secondary Aim 2. Not an identifier in isolation for a "
                           "5-digit ZIP, but bin into a coarser grouping before publication.",
            "Identifier?": "n",
        },
    ]

    # --- screening / triage --------------------------------------------
    fields += [
        {
            "Variable / Field Name": "asq_result",
            "Form Name": "screening_triage",
            "Section Header": "ASQ & Universal Triage",
            "Field Type": "radio",
            "Field Label": "ASQ screen result",
            "Choices, Calculations, OR Slider Labels": "positive, Positive | negative, Negative",
            "Identifier?": "n",
            "Required Field?": "y",
        },
        {
            "Variable / Field Name": "bypass_reason",
            "Form Name": "screening_triage",
            "Field Type": "radio",
            "Field Label": "Documented reason ASQ/PHQ-A/C-SSRS was not administered",
            "Choices, Calculations, OR Slider Labels":
                "altered_mental_status, Altered mental status | acuity_1, ESI acuity 1 | "
                "trauma, Acute trauma | cognitive_disability, Cognitive disability",
            "Branching Logic (Show field only if...)": "[asq_result] = ''",
            "Identifier?": "n",
        },
        {
            "Variable / Field Name": "presenting_complaint",
            "Form Name": "screening_triage",
            "Field Type": "radio",
            "Field Label": "Primary presenting complaint",
            "Choices, Calculations, OR Slider Labels":
                "psychiatric, Psychiatric | medical, Medical/somatic | trauma, Trauma",
            "Identifier?": "n",
            "Required Field?": "y",
        },
        {
            "Variable / Field Name": "track",
            "Form Name": "screening_triage",
            "Field Type": "radio",
            "Field Label": "Clinical track",
            "Choices, Calculations, OR Slider Labels":
                "behavioral, Behavioral Health Track | physical, Physical Health Track",
            "Branching Logic (Show field only if...)": "[asq_result] = 'positive'",
            "Identifier?": "n",
        },
        {
            "Variable / Field Name": "esi_level",
            "Form Name": "screening_triage",
            "Field Type": "radio",
            "Field Label": "Emergency Severity Index (ESI) level",
            "Choices, Calculations, OR Slider Labels": "1, 1 | 2, 2 | 3, 3 | 4, 4 | 5, 5",
            "Identifier?": "n",
        },
        {
            "Variable / Field Name": "cssrs_triage_result",
            "Form Name": "screening_triage",
            "Field Type": "radio",
            "Field Label": "C-SSRS Primary Care Screener triage result",
            "Choices, Calculations, OR Slider Labels": "1, Low risk | 2, Moderate risk | 3, High risk",
            "Branching Logic (Show field only if...)": "[asq_result] = 'positive'",
            "Identifier?": "n",
        },
    ]

    # --- clinical assessments (BH track: standard care; PH track: research add-on) --
    fields += [
        {
            "Variable / Field Name": "phqa_total",
            "Form Name": "clinical_assessments",
            "Section Header": "Standard Care / Research Add-On Scores",
            "Field Type": "text",
            "Field Label": "PHQ-A total score",
            "Text Validation Type OR Show Slider Number": "integer",
            "Text Validation Min": "0",
            "Text Validation Max": "27",
            "Branching Logic (Show field only if...)": "[asq_result] = 'positive'",
            "Identifier?": "n",
        },
        {
            "Variable / Field Name": "gad7_total",
            "Form Name": "clinical_assessments",
            "Field Type": "text",
            "Field Label": "GAD-7 total score",
            "Field Note": "Descriptive only -- never enters the confirmatory Step 1-4 model.",
            "Text Validation Type OR Show Slider Number": "integer",
            "Text Validation Min": "0",
            "Text Validation Max": "21",
            "Branching Logic (Show field only if...)": "[asq_result] = 'positive'",
            "Identifier?": "n",
        },
        {
            "Variable / Field Name": "cssrs_ideation_severity",
            "Form Name": "clinical_assessments",
            "Field Type": "radio",
            "Field Label": "C-SSRS Ideation Severity Score",
            "Choices, Calculations, OR Slider Labels":
                "1, Wish to be dead | 2, Non-specific active suicidal thoughts | "
                "3, Active suicidal ideation with any methods (not plan) | "
                "4, Active suicidal ideation with some intent to act | "
                "5, Active suicidal ideation with specific plan and intent",
            "Field Note": "Primary outcome.",
            "Branching Logic (Show field only if...)": "[asq_result] = 'positive'",
            "Identifier?": "n",
        },
        {
            "Variable / Field Name": "bssa_tier",
            "Form Name": "clinical_assessments",
            "Field Type": "radio",
            "Field Label": "BSSA disposition/tier",
            "Choices, Calculations, OR Slider Labels": "low, Low | moderate, Moderate | high, High",
            "Branching Logic (Show field only if...)": "[asq_result] = 'positive'",
            "Identifier?": "n",
        },
        {
            "Variable / Field Name": "med_risk_flag",
            "Form Name": "clinical_assessments",
            "Field Type": "yesno",
            "Field Label": "Documented elevated-risk medication/substance use "
                           "(e.g. opioids, benzodiazepines/sedative-hypnotics, "
                           "montelukast, isotretinoin, sertraline)",
            "Identifier?": "n",
        },
        {
            "Variable / Field Name": "comorbid_dx",
            "Form Name": "clinical_assessments",
            "Field Type": "radio",
            "Field Label": "Comorbid psychiatric diagnosis",
            "Choices, Calculations, OR Slider Labels":
                "anxiety, Anxiety | depression, Depression | none, None documented | other, Other",
            "Field Note": "Single primary comorbidity for pipeline compatibility -- if a patient "
                           "has several documented, record the clinically primary one here and the "
                           "rest in the free-text chart note, not in this field.",
            "Identifier?": "n",
        },
        {
            "Variable / Field Name": "icd10",
            "Form Name": "clinical_assessments",
            "Field Type": "text",
            "Field Label": "Primary ICD-10 code",
            "Identifier?": "n",
        },
    ]

    # --- inter-rater reliability (10% dual-abstraction subsample) ------
    fields += [
        {
            "Variable / Field Name": "presenting_complaint_2nd",
            "Form Name": "chart_abstraction_qc",
            "Section Header": "Dual Abstraction (10% subsample only)",
            "Field Type": "radio",
            "Field Label": "Presenting complaint -- second abstractor",
            "Choices, Calculations, OR Slider Labels":
                "psychiatric, Psychiatric | medical, Medical/somatic | trauma, Trauma",
            "Field Note": "Leave blank unless this chart was selected for dual abstraction.",
            "Identifier?": "n",
        },
        {
            "Variable / Field Name": "comorbid_dx_2nd",
            "Form Name": "chart_abstraction_qc",
            "Field Type": "radio",
            "Field Label": "Comorbid diagnosis -- second abstractor",
            "Choices, Calculations, OR Slider Labels":
                "anxiety, Anxiety | depression, Depression | none, None documented | other, Other",
            "Field Note": "Leave blank unless this chart was selected for dual abstraction.",
            "Identifier?": "n",
        },
    ]

    return fields


def generate(output_filename):
    with open(output_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        for row in build_fields():
            writer.writerow({h: row.get(h, "") for h in HEADERS})
    print(f"Wrote {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the RCHOC REDCap Data Dictionary (non-INQ fields)")
    parser.add_argument("--output", default="rchoc_data_dictionary.csv")
    args = parser.parse_args()
    generate(args.output)
