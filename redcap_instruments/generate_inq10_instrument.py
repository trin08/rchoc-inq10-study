"""
Generates the standalone INQ-10 REDCap instrument: the pre-disclosure
text as a descriptive field, followed by the 10 items as a matrix.

Item wording and order are taken from the protocol's own Table 1
(Hill et al., 2015) appendix reproduction, in the same original-item
order config.py assumes by default (1, 3, 9, 12, 14, 17, 19, 20, 21,
24). Field names are inq10_item_1 ... inq10_item_10, matching
config.INQ10_ITEM_COLUMNS exactly -- a REDCap export built from this
instrument needs no field-name edits before analyze_redcap.py can
score it.

Reverse-scoring (items 17, 19, 24) happens downstream in the analysis
pipeline, not here -- the survey shows each item in its natural,
un-reversed wording, which is standard practice for a Likert
instrument.

    python3 generate_inq10_instrument.py --output inq10_instrument.csv
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

FORM_NAME = "inq10_questionnaire"

DISCLOSURE_TEXT = (
    "Before you begin: The next 10 questions are part of a research study, separate from "
    "your medical care today. Doctors already ask about mood and thoughts of suicide to "
    "understand how you're doing. This study is looking at two other feelings that sometimes "
    "come up when people are having a hard time: feeling like a burden to others, and feeling "
    "disconnected from others. These questions help researchers understand whether those "
    "feelings add anything useful to what doctors already screen for, so that care can be "
    "better matched to what a person is actually going through. Your answers here will not "
    "change your care today, and you can stop at any time without any effect on your "
    "treatment. A member of the study team is here and can answer any questions before, "
    "during, or after."
)

INQ_CHOICES = "1, Not at all true for me | 2, 2 | 3, 3 | 4, 4 | 5, 5 | 6, 6 | 7, Very true for me"

# (original item number, wording) -- in default presentation order.
# Original numbering is what config.INQ10_FIELD_TO_ORIGINAL_ITEM maps
# against; keep this order in sync if you ever reorder the survey.
INQ10_ITEMS = [
    (1, "These days, the people in my life would be better off if I were gone"),
    (3, "These days, the people in my life would be happier without me"),
    (9, "These days, I think my death would be a relief to the people in my life"),
    (12, "These days, I think the people in my life wish they could be rid of me"),
    (14, "These days, I think I make things worse for the people in my life"),
    (17, "These days, I feel like I belong"),
    (19, "These days, I am fortunate to have many caring and supportive friends"),
    (20, "These days, I feel disconnected from other people"),
    (21, "These days, I often feel like an outsider in social gatherings"),
    (24, "These days, I am close to other people"),
]


def build_fields():
    fields = [{
        "Variable / Field Name": "inq10_disclosure",
        "Form Name": FORM_NAME,
        "Field Type": "descriptive",
        "Field Label": DISCLOSURE_TEXT,
        "Identifier?": "n",
    }]

    for position, (orig_item, wording) in enumerate(INQ10_ITEMS, start=1):
        fields.append({
            "Variable / Field Name": f"inq10_item_{position}",
            "Form Name": FORM_NAME,
            "Section Header": "Interpersonal Needs Questionnaire" if position == 1 else "",
            "Field Type": "radio",
            "Field Label": wording,
            "Choices, Calculations, OR Slider Labels": INQ_CHOICES,
            "Field Note": f"INQ original item {orig_item}",
            "Identifier?": "n",
            "Required Field?": "y",
            "Matrix Group Name": "inq10_matrix",
        })

    return fields


def generate(output_filename):
    with open(output_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        for row in build_fields():
            writer.writerow({h: row.get(h, "") for h in HEADERS})
    print(f"Wrote {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the standalone INQ-10 REDCap instrument")
    parser.add_argument("--output", default="inq10_instrument.csv")
    args = parser.parse_args()
    generate(args.output)
