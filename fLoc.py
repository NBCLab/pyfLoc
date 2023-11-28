"""Run the PsychoPy implementation of the fLoc task."""
import os
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from psychopy import core, event, logging, visual
from psychopy.constants import STARTED, STOPPED
from psychopy.gui import DlgFromDict
from yaml import Loader, load


def allocate_responses(events_df, response_times, response_window=1.0):
    """Assign responses to task trials.

    Parameters
    ----------
    events_df : :obj:`pandas.DataFrame`
        Initial dataframe containing information about trials, but not participant responses.
    response_times : :obj:`list` of :obj:`float` or None
        A list of times at which the participant responsed within the run.
    response_window : :obj:`float`
        The time after each trial's onset in which to accept a response.

    Returns
    -------
    events_df : :obj:`pandas.DataFrame`
        Updated dataframe with columns "response_time", "accuracy", and "classification" added.
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
        # Looping backwards lets us keep earliest response for RT.
        # Any response is *the* response, so the actual button doesn't matter.
        for rt in response_times[::-1]:
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
        for rt in response_times[::-1]:
            if onset <= rt < next_onset:
                # Ignore response window and use current trial's duration only,
                # since we really don't know which trial elicited the false positive.
                events_df.loc[trial_idx, "accuracy"] = 0
                events_df.loc[trial_idx, "classification"] = "false_positive"
                events_df.loc[trial_idx, "response_time"] = rt - onset
    return events_df


def randomize_carefully(elems, n_repeat=2):
    """Shuffle without consecutive duplicates.

    Parameters
    ----------
    elems : :obj:`list`
        List of unique elements from which to build the random list.
    n_repeat : :obj:`int`, optional
        Number of repeats of each element in ``elems`` to have in the output list.
        Default is 2.

    Returns
    -------
    res : :obj:`list`
        List of shuffled elements.

    Notes
    -----
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
    """Close window if escape is pressed.

    Parameters
    ----------
    win : :obj:`psychopy.visual.Window`
        Window to close.
    """
    if "escape" in event.getKeys():
        win.close()
        core.quit()


def draw_countdown(win, stim, duration):
    """Draw a countdown by the second.

    Parameters
    ----------
    win : :obj:`psychopy.visual.Window`
        Window in which to draw the countdown.
    stim : :obj:`psychopy.visual.TextStim`
    duration : :obj:`int`
        Number of seconds for which to draw the countdown.
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


def draw_until_keypress(win, stim, continue_keys=["5"], debug=False):
    """Draw a screen until a specific key is pressed.

    Parameters
    ----------
    win : :obj:`psychopy.visual.Window`
        Window in which to draw the stimulus.
    stim : :obj:`psychopy.visual.TextStim`
        Text stimulus (e.g., instructions) to draw until one of the ``continue_keys`` are pressed.
    continue_keys : :obj:`list` of :obj:`str`, optional
        Keys to accept to stop drawing the stimulus.
        Default is ["5"].
    debug : :obj:`bool`
        If True, then the screen will just wait 5 seconds and then continue.
        Default is False.
    """
    response = event.BuilderKeyResponse()
    win.callOnFlip(response.clock.reset)
    event.clearEvents(eventType="keyboard")
    if debug:
        time.wait(5)
        return

    while True:
        if isinstance(stim, list):
            for s in stim:
                s.draw()
        else:
            stim.draw()
        keys = event.getKeys(keyList=continue_keys)
        if any([ck in keys for ck in continue_keys]):
            return
        close_on_esc(win)
        win.flip()


def draw(win, stim, duration, clock):
    """Draw stimulus for a given duration.

    Parameters
    ----------
    win : :obj:`psychopy.visual.Window`
        Window in which to draw the stimulus.
    stim : object with ``.draw()`` method or list of such objects
        Stimulus to draw for desired duration.
    duration : :obj:`float`
        Duration in seconds to display the stimulus.
    clock : :obj:`psychopy.core.Clock`
        Clock object with which to track the duration.

    Notes
    -----
    According to the PsychoPy documentation, it would be more accurate to use a frame count than a
    time duration.
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

        # Allow any of keys 1 - 4
        keys = event.getKeys(keyList=["1", "2", "3", "4"], timeStamped=clock)
        if keys:
            response.keys.extend(keys)
            response.rt.append(response.clock.getTime())

        close_on_esc(win)
        win.flip()

    response.status = STOPPED
    return response.keys, response.rt


def main(debug=False):
    """Run the fLoc task."""
    # Ensure that relative paths start from the same directory as this script
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__)).decode(
            sys.getfilesystemencoding()
        )
    except AttributeError:
        script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load configuration file
    config_file = os.path.join(script_dir, "config.yml")
    with open(config_file, "r") as fo:
        config = load(fo, Loader=Loader)

    constants = config["constants"]
    trial_duration = constants["IMAGE_DURATION"] + constants["TARGET_ISI"]

    # Collect user input
    # ------------------
    # Remember to turn fullscr to True for the real deal.
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

    if not debug:
        exp_info = {
            "Subject": "",
            "Session": "",
            "Task": ["OneBack", "TwoBack", "Oddball"],
            "Image Set": ["default", "alternate", "both"],
            "Number of Runs": "4",
        }
        dlg = DlgFromDict(
            exp_info,
            title="Functional localizer",
            order=["Subject", "Session", "Task", "Image Set", "Number of Runs"],
        )
        if not dlg.OK:
            # Quit if user presses "Cancel" or "Close"
            core.quit()
    else:
        exp_info = {
            "Subject": "01",
            "Session": "01",
            "Task": "Oddball",
            "Image Set": "default",
            "Number of Runs": "1",
        }

    output_dir = os.path.join(script_dir, "data")
    os.makedirs(output_dir, exist_ok=True)

    # Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    ses_str = ""
    if exp_info["Session"]:
        ses_str = f"ses-{exp_info['Session'].zfill(2)}_"

    base_name = (
        f"sub-{exp_info['Subject'].zfill(2)}_{ses_str}task-localizer{exp_info['Task']}"
    )

    # save a log file for detail verbose info
    filename = os.path.join(output_dir, f"{base_name}_events")
    logging.LogFile(f"{filename}.log", level=logging.EXP)
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

    # Generate instruction stimulus
    if exp_info["Task"] == "Oddball":
        instruction_text = "Fixate.\nPress a button when a scrambled image appears."
    elif exp_info["Task"] == "TwoBack":
        instruction_text = (
            "Fixate.\nPress a button when an image repeats with one intervening image."
        )
    else:
        instruction_text = (
            "Fixate.\nPress a button when an image repeats on sequential trials."
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

    # Base stimulus image (the actual image is swapped out at each trial)
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

    # Fixation stimulus
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

    # Average performance across run to be shown between runs
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
                glob(os.path.join(script_dir, f"stimuli/{stimulus_folder}/*.jpg"))
                for stimulus_folder in stimulus_folders[category]
            ]
            # Unravel list of lists and clean up paths
            stimulus_files = [
                Path(item).as_posix() for sublist in stimulus_files for item in sublist
            ]
            stimuli[category] = stimulus_files
        else:
            # TODO: Support stimulus for baseline trials
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
    task_blocks = grabber_list * n_dupes

    # Scanner runtime
    # ---------------
    global_clock = core.Clock()  # to track the time since experiment started
    run_clock = core.Clock()  # to track time since each run starts (post scanner pulse)
    block_clock = core.Clock()  # to track duration of each block
    trial_clock = core.Clock()  # to track duration of each trial
    fixation_trial_clock = (
        core.Clock()
    )  # to account for fixation time spent loading image

    columns = [
        "onset",
        "duration",
        "trial_type",
        "miniblock_number",
        "category",
        "subcategory",
        "stim_file",
    ]
    # unnecessary, since run_frame is defined at end of for loop, but satisfies linter
    run_frame = None
    for i_run in range(n_runs):
        run_data = {c: [] for c in columns}
        run_label = i_run + 1
        events_file = os.path.join(
            output_dir, f"{base_name}_run-{run_label:02d}_events.tsv"
        )

        block_categories = randomize_carefully(
            standard_categories, n_blocks_per_category
        )
        np.random.shuffle(task_blocks)

        # Scanner runtime
        # ---------------
        # Wait for trigger from scanner to start run.
        if i_run == 0:
            # Show instructions for the first run until the scanner trigger
            draw_until_keypress(win=window, stim=instruction_text_box, debug=debug)
        else:
            # Show performance from the last run until the scanner trigger
            hit_count = (run_frame["classification"] == "true_positive").sum()
            n_probes = (
                run_frame["classification"]
                .isin(["false_negative", "true_positive"])
                .sum()
            )
            hit_rate = hit_count / n_probes
            fa_count = (run_frame["classification"] == "false_positive").sum()
            performance_str = (
                f"Hits: {hit_count}/{n_probes} ({hit_rate:.02f}%)\n"
                f"False alarms: {fa_count}"
            )
            performance_screen.setText(performance_str)
            performance_screen.draw()
            draw_until_keypress(win=window, stim=performance_screen, debug=debug)

        run_clock.reset()

        # Show countdown at beginning of run
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

        run_response_times = []
        nonbaseline_block_counter = 0
        for j_block, category in enumerate(block_categories):
            block_clock.reset()
            if category == "baseline":
                onset_time = run_clock.getTime()
                responses, _ = draw(
                    win=window,
                    stim=fixation,
                    duration=constants["N_STIMULI_PER_BLOCK"] * trial_duration,
                    clock=run_clock,
                )
                # The first element of each sublist is the actual button pressed,
                # but we don't care about that.
                run_response_times += [resp[1] for resp in responses]
                target_idx = None

                # Log info
                run_data["onset"].append(onset_time)
                run_data["duration"].append(block_clock.getTime())
                run_data["trial_type"].append("baseline")
                run_data["stim_file"].append("n/a")
                run_data["category"].append("baseline")
                run_data["subcategory"].append("baseline")
                run_data["miniblock_number"].append(j_block + 1)
            else:
                # Block of stimuli
                block_stimuli = list(
                    np.random.choice(
                        stimuli[category],
                        size=constants["N_STIMULI_PER_BLOCK"],
                        replace=False,
                    )
                )
                if task_blocks[nonbaseline_block_counter] == 1:
                    # Check for last block's target to make sure that two targets don't
                    # occur within the same response window
                    if (j_block > 0) and (target_idx is not None):
                        last_target_onset = (
                            ((constants["N_STIMULI_PER_BLOCK"] + 1) - target_idx)
                            * trial_duration
                            * -1
                        )
                        last_target_rw_offset = (
                            last_target_onset + constants["RESPONSE_WINDOW"]
                        )
                        first_viable_trial = int(
                            np.ceil(last_target_rw_offset / trial_duration)
                        )
                        first_viable_trial = np.maximum(0, first_viable_trial)
                        first_viable_trial += 1  # just to give it a one-trial buffer
                    else:
                        first_viable_trial = 0

                    # Adjust stimuli based on task
                    if exp_info["Task"] == "Oddball":
                        # target is scrambled image
                        target_idx = np.random.randint(
                            first_viable_trial, len(block_stimuli)
                        )
                        block_stimuli[target_idx] = np.random.choice(
                            stimuli["scrambled"]
                        )
                    elif exp_info["Task"] == "OneBack":
                        # target is second stim of same kind
                        first_viable_trial = np.maximum(first_viable_trial, 1)
                        target_idx = np.random.randint(
                            first_viable_trial, len(block_stimuli)
                        )
                        block_stimuli[target_idx] = block_stimuli[target_idx - 1]
                    elif exp_info["Task"] == "TwoBack":
                        # target is second stim of same kind
                        first_viable_trial = np.maximum(first_viable_trial, 2)
                        target_idx = np.random.randint(
                            first_viable_trial, len(block_stimuli)
                        )
                        block_stimuli[target_idx] = block_stimuli[target_idx - 2]
                else:
                    target_idx = None

                for k_stim, stim_file in enumerate(block_stimuli):
                    fixation_trial_clock.reset()
                    stim_image.image = stim_file
                    trial_clock.reset()
                    onset_time = run_clock.getTime()

                    # Draw the stimulus.
                    # Accept responses during the stimulus presentation.
                    responses, _ = draw(
                        win=window,
                        stim=[stim_image, fixation],
                        duration=constants["IMAGE_DURATION"],
                        clock=run_clock,
                    )
                    run_response_times += [resp[1] for resp in responses]

                    # Log the inter-stimulus interval.
                    duration = trial_clock.getTime()
                    loading_plus_stim_duration = fixation_trial_clock.getTime()
                    isi_dur = np.maximum(trial_duration - loading_plus_stim_duration, 0)

                    # Draw the post-stimulus fixation.
                    # Accept responses during this fixation presentation.
                    responses, _ = draw(
                        win=window, stim=fixation, duration=isi_dur, clock=run_clock
                    )

                    run_response_times += [resp[1] for resp in responses]
                    relative_stim_file = os.path.relpath(stim_file)
                    subcategory = os.path.basename(relative_stim_file).split("-")[0]

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
                    run_data["miniblock_number"].append(j_block + 1)

                nonbaseline_block_counter += 1

            # Unused duration
            # block_duration = block_clock.getTime()

        run_frame = pd.DataFrame(run_data)
        run_frame = allocate_responses(
            run_frame,
            run_response_times,
            response_window=constants["RESPONSE_WINDOW"],
        )
        run_frame.to_csv(
            events_file,
            sep="\t",
            lineterminator="\n",
            na_rep="n/a",
            index=False,
            float_format="%.2f",
        )

        # Last fixation
        last_iti = constants["TOTAL_DURATION"] - run_clock.getTime()
        draw(win=window, stim=fixation, duration=last_iti, clock=run_clock)

        print(f"Total duration of run: {run_clock.getTime()}")

    # Show the final run's performance
    # Scanner is off for this
    hit_count = (run_frame["classification"] == "true_positive").sum()
    n_probes = (
        run_frame["classification"].isin(["false_negative", "true_positive"]).sum()
    )
    hit_rate = hit_count / n_probes
    fa_count = (run_frame["classification"] == "false_positive").sum()
    performance_str = (
        f"Hits: {hit_count}/{n_probes} ({hit_rate:.02f}%)\nFalse alarms: {fa_count}"
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


if __name__ == "__main__":
    main(debug=False)
