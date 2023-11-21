import json

events_description = {
    "miniblock_number": {
        "LongName": "miniblock number",
        "Description": "Index of the block to which each trial belongs",
    },
    "category": {
        "LongName": "category",
        "Description": "Overall stimulus category",
        "Levels": {
            "bodies": "",
            "characters": "",
            "faces": "",
            "objects": "",
            "places": "",
        },
    },
    "subcategory": {
        "LongName": "subcategory",
        "Description": 'Subtype of the overall category (e.g., "car" under "objects").',
    },
    "accuracy": {
        "LongName": "accuracy",
        "Description": (
            "Participant performance. Trials are automatically "
            "correct unless there is a false alarm or if the trial "
            "is a task trial and the participant gets it wrong."
        ),
        "Levels": {"0": "Incorrect", "1": "Correct"},
    },
    "classification": {
        "LongName": "performance classification",
        "Description": "How trial accuracy is labeled.",
        "Levels": {
            "true_positive": (
                "A correct button-press on a task target trial. "
                'Also referred to as a "hit".'
            ),
            "false_positive": (
                "An incorrect button-press on a baseline trial. "
                'Also referred to as a "false alarm".'
            ),
            "false_negative": (
                "An incorrect non-press on a task target trial. "
                'Also referred to as a "miss".'
            ),
            "true_negative": ("A correct non-press on a baseline trial."),
        },
    },
}

for task_type in ["Oddball", "Two-Back", "One-Back"]:
    bold_description = {
        "CogAtlasID": "trm_553e85265f51e",
        "TaskName": "dual functional localizer/{0}".format(task_type.lower()),
    }

    with open(
        "task-localizer{0}_events.json".format(task_type.replace("-", "")), "w"
    ) as fo:
        json.dump(events_description, fo, sort_keys=True, indent=4)

    with open(
        "task-localizer{0}_bold.json".format(task_type.replace("-", "")), "w"
    ) as fo:
        json.dump(bold_description, fo, sort_keys=True, indent=4)
