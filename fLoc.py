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
    events_df.loc[events_df["trial_type"] == "category", "classification"] = "true_negative"
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


def prepare_trials(
    stimulus_categories,
    stimuli,
    constants,
    task,
):
    """Define the order and stimuli for the trials in a single run."""
    trial_duration = constants["IMAGE_DURATION"] + constants["TARGET_ISI"]
    n_categories = len(stimulus_categories)
    n_blocks_per_category = int(np.floor(constants["N_BLOCKS"] / n_categories))

    # Determine which trials will be task
    # This might be overly convoluted, but it maximizes balance between
    # task/non-task instead of just sampling with set probabilities
    nontask_rate = 1 - constants["TASK_RATE"]
    task_mult = 1 / np.minimum(constants["TASK_RATE"], nontask_rate)
    n_task_prop = int(task_mult * constants["TASK_RATE"])
    n_nontask_prop = int(task_mult * nontask_rate)
    grabber_list = [1] * n_task_prop + [0] * n_nontask_prop

    # We want to ensure that tasks are not assigned to baseline blocks
    n_nonbaseline_blocks = int(constants["N_BLOCKS"] * (n_categories - 1) / n_categories)
    n_dupes = int(np.ceil(n_nonbaseline_blocks / len(grabber_list)))
    task_blocks = grabber_list * n_dupes

    block_categories = randomize_carefully(stimulus_categories, n_blocks_per_category)
    np.random.shuffle(task_blocks)
    task_blocks_full = np.zeros(len(block_categories))
    task_blocks_full[np.array(block_categories) != "baseline"] = task_blocks

    run_config = {
        "stimuli": ["n/a"],
        "trial_type": ["countdown"],
        "category": ["n/a"],
        "subcategory": ["n/a"],
        "miniblock_number": ["n/a"],
        "expected_duration": [constants["COUNTDOWN_DURATION"]],
    }
    target_trial_idx = None
    for j_block, category in enumerate(block_categories):
        if category == "baseline":
            run_config["stimuli"].append("n/a")
            run_config["trial_type"].append("baseline")
            run_config["category"].append("n/a")
            run_config["subcategory"].append("n/a")
            run_config["expected_duration"].append(
                constants["N_STIMULI_PER_BLOCK"] * trial_duration
            )
            run_config["miniblock_number"].append(j_block)
        else:
            n_trials_in_block = constants["N_STIMULI_PER_BLOCK"]
            # Block of stimuli
            block_stimuli = list(
                np.random.choice(
                    stimuli[category],
                    size=n_trials_in_block,
                    replace=False,
                )
            )
            block_subcategories = [os.path.basename(s).split("-")[0] for s in block_stimuli]
            run_config["stimuli"] += block_stimuli
            run_config["trial_type"] += ["category"] * n_trials_in_block
            run_config["category"] += [category] * n_trials_in_block
            run_config["subcategory"] += block_subcategories
            run_config["miniblock_number"] += [j_block] * n_trials_in_block
            run_config["expected_duration"] += [trial_duration] * n_trials_in_block

            if task_blocks_full[j_block] == 1:
                start_of_block = len(run_config["trial_type"]) - n_trials_in_block
                # Check for last block's target to make sure that two targets don't
                # occur within the same response window
                if (j_block > 0) and (target_trial_idx is not None):
                    last_target_onset = np.sum(run_config["expected_duration"][:target_trial_idx])
                    last_target_rw_offset = last_target_onset + constants["RESPONSE_WINDOW"]

                    first_viable_trial = None
                    for k_trial, trial_offset in enumerate(range(n_trials_in_block + 1, 1, -1)):
                        onset = np.sum(run_config["expected_duration"][:-trial_offset])
                        if onset > last_target_rw_offset:
                            first_viable_trial = k_trial
                            break

                else:
                    first_viable_trial = 0

                # Adjust stimuli based on task
                if task == "Oddball":
                    # target is scrambled image
                    target_trial_idx = np.random.randint(
                        start_of_block + first_viable_trial,
                        start_of_block + n_trials_in_block,
                    )
                    run_config["stimuli"][target_trial_idx] = np.random.choice(
                        stimuli["scrambled"]
                    )
                elif task == "OneBack":
                    # target is second stim of same kind
                    first_viable_trial = np.maximum(first_viable_trial, 1)
                    target_trial_idx = np.random.randint(
                        start_of_block + first_viable_trial,
                        start_of_block + n_trials_in_block,
                    )
                    run_config["stimuli"][target_trial_idx] = run_config["stimuli"][
                        target_trial_idx - 1
                    ]
                elif task == "TwoBack":
                    # target is second stim of same kind
                    first_viable_trial = np.maximum(first_viable_trial, 2)
                    target_trial_idx = np.random.randint(
                        start_of_block + first_viable_trial,
                        start_of_block + n_trials_in_block,
                    )
                    run_config["stimuli"][target_trial_idx] = run_config["stimuli"][
                        target_trial_idx - 2
                    ]

                run_config["trial_type"][target_trial_idx] = task.lower()

    return run_config


def main(debug=False):
    """Run the fLoc task."""
    from psychopy.gui import DlgFromDict

    # Ensure that relative paths start from the same directory as this script
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__)).decode(sys.getfilesystemencoding())
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
        fullscr=False,
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

    base_name = f"sub-{exp_info['Subject'].zfill(2)}_{ses_str}task-localizer{exp_info['Task']}"

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
        instruction_text = "Fixate.\nPress a button when an image repeats on sequential trials."
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

    # Scanner runtime
    # ---------------
    global_clock = core.Clock()  # to track the time since experiment started
    run_clock = core.Clock()  # to track time since each run starts (post scanner pulse)
    trial_clock = core.Clock()  # to track duration of each trial
    fixation_trial_clock = core.Clock()  # to account for fixation time spent loading image

    # unnecessary, since run_frame is defined at end of for loop, but satisfies linter
    run_frame = None
    for i_run in range(n_runs):
        run_label = i_run + 1
        events_file = os.path.join(output_dir, f"{base_name}_run-{run_label:02d}_events.tsv")

        run_data = prepare_trials(
            stimulus_categories=standard_categories,
            stimuli=stimuli,
            task=exp_info["Task"],
            constants=config["constants"],
        )
        run_data["onset"] = []
        run_data["duration"] = []

        # Scanner runtime
        # ---------------
        # Wait for trigger from scanner to start run.
        if i_run == 0:
            # Show instructions for the first run until the scanner trigger
            draw_until_keypress(win=window, stim=instruction_text_box, debug=debug)
        else:
            # Show performance from the last run until the scanner trigger
            hit_count = (run_frame["classification"] == "true_positive").sum()
            n_probes = run_frame["classification"].isin(["false_negative", "true_positive"]).sum()
            hit_rate = hit_count / n_probes
            fa_count = (run_frame["classification"] == "false_positive").sum()
            performance_str = (
                f"Hits: {hit_count}/{n_probes} ({hit_rate:.02f}%)\n" f"False alarms: {fa_count}"
            )
            performance_screen.setText(performance_str)
            performance_screen.draw()
            draw_until_keypress(win=window, stim=performance_screen, debug=debug)

        run_clock.reset()
        run_response_times = []
        for i_trial in range(len(run_data["trial_type"])):
            trial_type = run_data["trial_type"][i_trial]
            stim_file = run_data["stimuli"][i_trial]
            trial_clock.reset()

            actual_onset_time = run_clock.getTime()
            if trial_type == "countdown":
                draw_countdown(
                    win=window,
                    stim=countdown_text_box,
                    duration=constants["COUNTDOWN_DURATION"],
                )

                run_data["onset"].append(actual_onset_time)
                run_data["duration"].append(trial_clock.getTime())

            elif trial_type == "baseline":
                responses, _ = draw(
                    win=window,
                    stim=fixation,
                    duration=constants["N_STIMULI_PER_BLOCK"] * trial_duration,
                    clock=run_clock,
                )
                # The first element of each sublist is the actual button pressed,
                # but we don't care about that.
                run_response_times += [resp[1] for resp in responses]

                run_data["onset"].append(actual_onset_time)
                run_data["duration"].append(trial_clock.getTime())

            else:
                fixation_trial_clock.reset()
                stim_image.image = stim_file

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
                # TODO: Try to adjust timing based on *expected* onset and duration here.
                actual_duration = trial_clock.getTime()
                loading_plus_stim_duration = fixation_trial_clock.getTime()
                isi_dur = np.maximum(trial_duration - loading_plus_stim_duration, 0)

                # Draw the post-stimulus fixation.
                # Accept responses during this fixation presentation.
                responses, _ = draw(
                    win=window,
                    stim=fixation,
                    duration=isi_dur,
                    clock=run_clock,
                )

                run_response_times += [resp[1] for resp in responses]

                # Log info
                run_data["onset"].append(actual_onset_time)
                run_data["duration"].append(actual_duration)

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
    n_probes = run_frame["classification"].isin(["false_negative", "true_positive"]).sum()
    hit_rate = hit_count / n_probes
    fa_count = (run_frame["classification"] == "false_positive").sum()
    performance_str = f"Hits: {hit_count}/{n_probes} ({hit_rate:.02f}%)\nFalse alarms: {fa_count}"
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
