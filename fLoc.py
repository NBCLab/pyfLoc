from __future__ import absolute_import, division, print_function

import json
import os
import os.path as op
import sys
import time
from glob import glob

import numpy as np
import pandas as pd
import serial
from psychopy import core, event, gui, logging, visual
from psychopy.constants import STARTED, STOPPED


def allocate_responses(events_df, responses, response_times, response_window=1):
    """
    Assign responses to task trials.
    """
    # Let's start by locating target trials
    task_types = ["oddball", "oneback", "twoback"]
    response_times = response_times[:]  # copy
    target_trial_idx = events_df["trial_type"].isin(task_types)
    nontarget_trial_idx = ~target_trial_idx

    events_df["response_time"] = "n/a"
    events_df["accuracy"] = "n/a"
    events_df["classification"] = "n/a"

    # Defaults
    events_df.loc[events_df["trial_type"] == "category", "classification"] = 1
    events_df.loc[
        events_df["trial_type"] == "category", "classification"
    ] = "true_negative"
    events_df.loc[target_trial_idx, "accuracy"] = 0  # default to miss
    events_df.loc[target_trial_idx, "classification"] = "false_negative"

    # Log hits
    for trial_idx in events_df.index[target_trial_idx]:
        onset = events_df.loc[trial_idx, "onset"]
        keep_idx = []
        # Looping backwards lets us keep earliest response for RT
        # Any response is *the* response, so the actual button doesn't matter.
        for i_resp, rt in enumerate(response_times[::-1]):
            if onset <= rt <= (onset + response_window):
                events_df.loc[trial_idx, "accuracy"] = 1
                events_df.loc[trial_idx, "response_time"] = rt - onset
                events_df.loc[trial_idx, "classification"] = "true_positive"
            else:
                keep_idx.append(response_times.index(rt))
        response_times = [response_times[i] for i in sorted(keep_idx)]

    # Log false alarms
    for trial_idx in events_df.index[nontarget_trial_idx]:
        onset = events_df.loc[trial_idx, "onset"]
        if trial_idx == events_df.index.values[-1]:
            next_onset = onset + response_window  # arbitrary duration
        else:
            next_onset = events_df.loc[trial_idx + 1, "onset"]

        # Looping backwards lets us keep earliest response for RT
        for i_resp, rt in enumerate(response_times[::-1]):
            if onset <= rt < next_onset:
                # Ignore response window and use current trial's duration only
                events_df.loc[trial_idx, "accuracy"] = 0
                events_df.loc[trial_idx, "classification"] = "false_positive"
                events_df.loc[trial_idx, "response_time"] = rt - onset
    return events_df


def randomize_carefully(elems, n_repeat=2):
    """
    Shuffle without consecutive duplicates
    From https://stackoverflow.com/a/22963275/2589328
    """
    s = set(elems)
    res = []
    for n in range(n_repeat):
        if res:
            # Avoid the last placed element
            lst = list(s.difference({res[-1]}))
            # Shuffle
            np.random.shuffle(lst)
            lst.append(res[-1])
            # Shuffle once more to avoid obvious repeating patterns in the last position
            lst[1:] = np.random.choice(lst[1:], size=len(lst) - 1, replace=False)
        else:
            lst = elems[:]
            np.random.shuffle(lst)
        res.extend(lst)
    return res


def close_on_esc(win):
    """
    Closes window if escape is pressed
    """
    if "escape" in event.getKeys():
        win.close()
        core.quit()


def draw_countdown(win, stim, duration):
    """
    Draw a countdown by the second
    """
    countdown_clock = core.Clock()
    countdown_sec = duration
    remaining_time = duration
    stim.setText(countdown_sec)
    while remaining_time > 0:
        stim.draw()
        close_on_esc(win)
        win.flip()
        remaining_time = duration - countdown_clock.getTime()
        if np.floor(remaining_time) <= countdown_sec:
            stim.setText(countdown_sec)
            countdown_sec -= 1


def draw_until_keypress(win, stim, continueKeys=["5"]):
    """ """
    response = event.BuilderKeyResponse()
    win.callOnFlip(response.clock.reset)
    event.clearEvents(eventType="keyboard")
    while True:
        if isinstance(stim, list):
            for s in stim:
                s.draw()
        else:
            stim.draw()
        keys = event.getKeys(keyList=continueKeys)
        if any([ck in keys for ck in continueKeys]):
            return
        close_on_esc(win)
        win.flip()


def draw(win, stim, duration, clock):
    """
    Draw stimulus for a given duration.

    Parameters
    ----------
    win : (visual.Window)
    stim : object with `.draw()` method or list of such objects
    duration : (numeric)
        duration in seconds to display the stimulus
    """
    # Use a busy loop instead of sleeping so we can exit early if need be.
    start_time = time.time()
    response = event.BuilderKeyResponse()
    response.tStart = start_time
    response.frameNStart = 0
    response.status = STARTED
    win.callOnFlip(response.clock.reset)
    event.clearEvents(eventType="keyboard")
    while time.time() - start_time < duration:
        if isinstance(stim, list):
            for s in stim:
                s.draw()
        else:
            stim.draw()
        keys = event.getKeys(keyList=["1", "2", "3", "4"], timeStamped=clock)
        if keys:
            response.keys.extend(keys)
            response.rt.append(response.clock.getTime())
        close_on_esc(win)
        win.flip()
    response.status = STOPPED
    return response.keys, response.rt


if __name__ == "__main__":
    # Ensure that relative paths start from the same directory as this script
    try:
        script_dir = op.dirname(op.abspath(__file__)).decode(
            sys.getfilesystemencoding()
        )
    except AttributeError:
        script_dir = op.dirname(op.abspath(__file__))

    # Load configuration file
    config_file = op.join(script_dir, "config.json")
    with open(config_file, "r") as fo:
        config = json.load(fo)
    constants = config["constants"]
    constants["TRIAL_DURATION"] = constants["IMAGE_DURATION"] + constants["TARGET_ISI"]

    # Collect user input
    # ------------------
    # Remember to turn fullscr to True for the real deal.
    exp_info = {
        "Subject": "",
        "Session": "",
        "Task": ["OneBack", "TwoBack", "Oddball"],
        "Image Set": ["default", "alternate", "both"],
        "Number of Runs": "4",
        "BioPac": ["Yes", "No"],
    }
    dlg = gui.DlgFromDict(
        exp_info,
        title="Functional localizer: {}".format(exp_info["Task"]),
        order=["Subject", "Session", "Task", "Image Set", "Number of Runs", "BioPac"],
    )
    window = visual.Window(
        fullscr=True,
        size=(800, 600),
        monitor="testMonitor",
        units="pix",
        allowStencil=False,
        allowGUI=False,
        color="gray",
        colorSpace="rgb",
        blendMode="avg",
        useFBO=True,
    )
    if not dlg.OK:
        core.quit()

    # Establish serial port connection
    if exp_info["BioPac"] == "Yes":
        ser = serial.Serial("COM2", 115200)

    if not op.exists(op.join(script_dir, "data")):
        os.makedirs(op.join(script_dir, "data"))

    # Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    base_name = "sub-{0}_ses-{1}_task-localizer{2}".format(
        exp_info["Subject"].zfill(2), exp_info["Session"].zfill(2), exp_info["Task"]
    )

    # save a log file for detail verbose info
    filename = op.join(script_dir, "data/{0}_events".format(base_name))
    logfile = logging.LogFile(filename + ".log", level=logging.EXP)
    logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

    # Initialize stimuli
    # ------------------
    countdown_text_box = visual.TextStim(
        win=window,
        name="countdown",
        text=None,
        font="Arial",
        height=50,
        pos=(0, 0),
        wrapWidth=30,
        ori=0,
        color="white",
        colorSpace="rgb",
        opacity=1,
        depth=-1.0,
    )
    if exp_info["Task"] == "Oddball":
        instruction_text = "Fixate. Press a button when a scrambled image appears."
    elif exp_info["Task"] == "TwoBack":
        instruction_text = (
            "Fixate. Press a button when an image repeats with one intervening image."
        )
    else:
        instruction_text = (
            "Fixate. Press a button when an image repeats on sequential trials."
        )
    instruction_text_box = visual.TextStim(
        win=window,
        name="instructions",
        text=instruction_text,
        font="Arial",
        height=50,
        pos=(0, 0),
        wrapWidth=900,
        ori=0,
        color="white",
        colorSpace="rgb",
        opacity=1,
        depth=-1.0,
    )
    stim_image = visual.ImageStim(
        win=window,
        name="stimulus",
        image=None,
        ori=0,
        pos=(0, 0),
        size=(768, 768),
        color=[1, 1, 1],
        colorSpace="rgb",
        opacity=1,
        depth=-1.0,
        interpolate=False,
    )
    fixation = visual.TextStim(
        win=window,
        name="fixation",
        text="\u2022",
        font="Arial",
        pos=(0, 0),
        height=30,
        wrapWidth=None,
        ori=0,
        color="red",
        colorSpace="rgb",
        opacity=1,
        depth=0.0,
    )
    performance_screen = visual.TextStim(
        win=window,
        name="performance_screen",
        text=None,
        font="Arial",
        pos=(0, 0),
        height=50,
        wrapWidth=None,
        ori=0,
        color="white",
        colorSpace="rgb",
        opacity=1,
        depth=0.0,
    )

    # Collect stimulus sets
    n_runs = int(exp_info["Number of Runs"])
    stimulus_folders = config["category_sets"][exp_info["Image Set"]]

    standard_categories = [cat for cat in stimulus_folders.keys() if cat != "scrambled"]
    n_categories = len(standard_categories)
    n_blocks_per_category = int(np.floor(constants["N_BLOCKS"] / n_categories))

    stimuli = {}
    for category in stimulus_folders.keys():
        if stimulus_folders[category] is not None:
            stimulus_files = [
                glob(op.join(script_dir, "stimuli/{}/*.jpg".format(stimulus_folder)))
                for stimulus_folder in stimulus_folders[category]
            ]
            # Unravel list of lists
            stimulus_files = [item for sublist in stimulus_files for item in sublist]
            # Clean up paths
            stimulus_files = [
                op.realpath(item).replace("\\", "/") for item in stimulus_files
            ]
            stimuli[category] = stimulus_files
        else:
            stimuli[category] = None  # baseline trials just have fixation

    # Determine which trials will be task
    # This might be overly convoluted, but it maximizes balance between
    # task/non-task instead of just sampling with set probabilities
    nontask_rate = 1 - constants["TASK_RATE"]
    task_mult = 1 / np.minimum(constants["TASK_RATE"], nontask_rate)
    n_task_prop = int(task_mult * constants["TASK_RATE"])
    n_nontask_prop = int(task_mult * nontask_rate)
    grabber_list = [1] * n_task_prop + [0] * n_nontask_prop

    # We want to ensure that tasks are not assigned to baseline blocks
    n_nonbaseline_blocks = int(
        constants["N_BLOCKS"] * (n_categories - 1) / n_categories
    )
    n_dupes = int(np.ceil(n_nonbaseline_blocks / len(grabber_list)))
    task_miniblocks = grabber_list * n_dupes

    # Scanner runtime
    # ---------------
    global_clock = core.Clock()  # to track the time since experiment started
    run_clock = core.Clock()  # to track time since each run starts (post scanner pulse)
    miniblock_clock = core.Clock()  # to track duration of each miniblock
    trial_clock = core.Clock()  # to track duration of each trial
    fixation_trial_clock = (
        core.Clock()
    )  # to account for fixation time spent loading image

    for i_run in range(n_runs):
        COLUMNS = [
            "onset",
            "duration",
            "trial_type",
            "miniblock_number",
            "category",
            "subcategory",
            "stim_file",
        ]
        run_data = {c: [] for c in COLUMNS}
        run_label = i_run + 1
        outfile = op.join(
            script_dir,
            "data",
            "{0}_run-{1:02d}_events.tsv".format(base_name, run_label),
        )

        miniblock_categories = randomize_carefully(
            standard_categories, n_blocks_per_category
        )
        np.random.shuffle(task_miniblocks)

        # Scanner runtime
        # ---------------
        # Wait for trigger from scanner.
        if i_run == 0:
            # Instructions for the first run
            draw_until_keypress(win=window, stim=instruction_text_box)
        else:
            # Performance for the rest of the runs
            hit_count = (run_frame["classification"] == "true_positive").sum()
            n_probes = (
                run_frame["classification"]
                .isin(["false_negative", "true_positive"])
                .sum()
            )
            hit_rate = hit_count / n_probes
            fa_count = (run_frame["classification"] == "false_positive").sum()
            performance_str = ("Hits: {0}/{1} ({2:.02f}%)\nFalse alarms: {3}").format(
                hit_count, n_probes, hit_rate, fa_count
            )
            performance_screen.setText(performance_str)
            performance_screen.draw()
            draw_until_keypress(win=window, stim=performance_screen)

        # Start recording
        if exp_info["BioPac"] == "Yes":
            ser.write("FF")

        run_clock.reset()

        # Show countdown
        draw_countdown(
            win=window,
            stim=countdown_text_box,
            duration=constants["COUNTDOWN_DURATION"],
        )

        real_countdown_duration = run_clock.getTime()
        run_data["onset"].append(0)
        run_data["duration"].append(real_countdown_duration)
        run_data["trial_type"].append("countdown")
        run_data["stim_file"].append("n/a")
        run_data["category"].append("n/a")
        run_data["subcategory"].append("n/a")
        run_data["miniblock_number"].append("n/a")

        run_responses, run_response_times = [], []
        nonbaseline_block_counter = 0
        for j_miniblock, category in enumerate(miniblock_categories):
            miniblock_clock.reset()
            if category == "baseline":
                onset_time = run_clock.getTime()
                responses, _ = draw(
                    win=window,
                    stim=fixation,
                    duration=(
                        constants["N_STIMULI_PER_BLOCK"] * constants["TRIAL_DURATION"]
                    ),
                    clock=run_clock,
                )
                run_responses += [resp[0] for resp in responses]
                run_response_times += [resp[1] for resp in responses]
                target_idx = None

                # Log info
                run_data["onset"].append(onset_time)
                run_data["duration"].append(miniblock_clock.getTime())
                run_data["trial_type"].append("baseline")
                run_data["stim_file"].append("n/a")
                run_data["category"].append("baseline")
                run_data["subcategory"].append("baseline")
                run_data["miniblock_number"].append(j_miniblock + 1)
            else:
                # Block of stimuli
                miniblock_stimuli = list(
                    np.random.choice(
                        stimuli[category],
                        size=constants["N_STIMULI_PER_BLOCK"],
                        replace=False,
                    )
                )
                if task_miniblocks[nonbaseline_block_counter] == 1:
                    # Check for last block's target to make sure that two targets don't
                    # occur within the same response window
                    if (j_miniblock > 0) and (target_idx is not None):
                        last_target_onset = (
                            ((constants["N_STIMULI_PER_BLOCK"] + 1) - target_idx)
                            * constants["TRIAL_DURATION"]
                            * -1
                        )
                        last_target_rw_offset = (
                            last_target_onset + constants["RESPONSE_WINDOW"]
                        )
                        first_viable_trial = int(
                            np.ceil(last_target_rw_offset / constants["TRIAL_DURATION"])
                        )
                        first_viable_trial = np.maximum(0, first_viable_trial)
                        first_viable_trial += 1  # just to give it a one-trial buffer
                    else:
                        first_viable_trial = 0

                    # Adjust stimuli based on task
                    if exp_info["Task"] == "Oddball":
                        # target is scrambled image
                        target_idx = np.random.randint(
                            first_viable_trial, len(miniblock_stimuli)
                        )
                        miniblock_stimuli[target_idx] = np.random.choice(
                            stimuli["scrambled"]
                        )
                    elif exp_info["Task"] == "OneBack":
                        # target is second stim of same kind
                        first_viable_trial = np.maximum(first_viable_trial, 1)
                        target_idx = np.random.randint(
                            first_viable_trial, len(miniblock_stimuli)
                        )
                        miniblock_stimuli[target_idx] = miniblock_stimuli[
                            target_idx - 1
                        ]
                    elif exp_info["Task"] == "TwoBack":
                        # target is second stim of same kind
                        first_viable_trial = np.maximum(first_viable_trial, 2)
                        target_idx = np.random.randint(
                            first_viable_trial, len(miniblock_stimuli)
                        )
                        miniblock_stimuli[target_idx] = miniblock_stimuli[
                            target_idx - 2
                        ]
                else:
                    target_idx = None

                for k_stim, stim_file in enumerate(miniblock_stimuli):
                    fixation_trial_clock.reset()
                    stim_image.image = stim_file
                    trial_clock.reset()
                    onset_time = run_clock.getTime()
                    responses, _ = draw(
                        win=window,
                        stim=[stim_image, fixation],
                        duration=constants["IMAGE_DURATION"],
                        clock=run_clock,
                    )
                    run_responses += [resp[0] for resp in responses]
                    run_response_times += [resp[1] for resp in responses]
                    duration = trial_clock.getTime()
                    loading_plus_stim_duration = fixation_trial_clock.getTime()
                    isi_dur = np.maximum(
                        constants["TRIAL_DURATION"] - loading_plus_stim_duration, 0
                    )
                    responses, _ = draw(
                        win=window, stim=fixation, duration=isi_dur, clock=run_clock
                    )

                    run_responses += [resp[0] for resp in responses]
                    run_response_times += [resp[1] for resp in responses]
                    relative_stim_file = op.sep.join(stim_file.split("/")[-2:])
                    subcategory = stim_file.split("/")[-2]

                    if k_stim == target_idx:
                        trial_type = exp_info["Task"].lower()
                    else:
                        trial_type = "category"

                    # Log info
                    run_data["onset"].append(onset_time)
                    run_data["duration"].append(duration)
                    run_data["trial_type"].append(trial_type)
                    run_data["stim_file"].append(relative_stim_file)
                    run_data["category"].append(category)
                    run_data["subcategory"].append(subcategory)
                    run_data["miniblock_number"].append(j_miniblock + 1)
                nonbaseline_block_counter += 1
            miniblock_duration = miniblock_clock.getTime()

        run_frame = pd.DataFrame(run_data)
        run_frame = allocate_responses(
            run_frame,
            run_responses,
            run_response_times,
            response_window=constants["RESPONSE_WINDOW"],
        )
        run_frame.to_csv(
            outfile,
            sep="\t",
            line_terminator="\n",
            na_rep="n/a",
            index=False,
            float_format="%.2f",
        )

        # Last fixation
        last_iti = constants["TOTAL_DURATION"] - run_clock.getTime()
        draw(win=window, stim=fixation, duration=last_iti, clock=run_clock)

        # End recording
        if exp_info["BioPac"] == "Yes":
            ser.write("00")

        print("Total duration of run: {}".format(run_clock.getTime()))
    # end run_loop

    # Shut down serial port connection
    if exp_info["BioPac"] == "Yes":
        ser.close()

    # Scanner is off for this
    hit_count = (run_frame["classification"] == "true_positive").sum()
    n_probes = (
        run_frame["classification"].isin(["false_negative", "true_positive"]).sum()
    )
    hit_rate = hit_count / n_probes
    fa_count = (run_frame["classification"] == "false_positive").sum()
    performance_str = ("Hits: {0}/{1} ({2:.02f}%)\nFalse alarms: {3}").format(
        hit_count, n_probes, hit_rate, fa_count
    )
    performance_screen.setText(performance_str)
    draw(
        win=window,
        stim=performance_screen,
        duration=constants["END_SCREEN_DURATION"],
        clock=global_clock,
    )
    window.flip()

    logging.flush()

    # make sure everything is closed down
    window.close()
    core.quit()
