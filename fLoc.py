from __future__ import absolute_import, division, print_function
import sys
import time
import serial
import os.path as op
from glob import glob

import numpy as np
import pandas as pd

from psychopy import gui, visual, core, data, event, logging
from psychopy.constants import STARTED, STOPPED

COUNTDOWN_DURATION = 12
N_STIMULI_PER_BLOCK = 12
IMAGE_DURATION = 0.4
TARGET_ISI = 0.1
TOTAL_DURATION = 240  # four minutes
END_SCREEN_DURATION = 2
N_BLOCKS = (TOTAL_DURATION - COUNTDOWN_DURATION) / (N_STIMULI_PER_BLOCK * (IMAGE_DURATION + TARGET_ISI))
N_BLOCKS = int(np.floor(N_BLOCKS))
TOTAL_DURATION = 252  # 4:12, giving time for lead-in and ending fixations
N_BLOCKS = 40
TASK_RATE = 0.5  # rate of actual tasks throughout scan. Should be specified as fraction
STIMULUS_HEIGHT = 1.  # height for images
RESPONSE_WINDOW = 1.  # time for participants to response to a target stimulus


def allocate_responses(responses, response_times, events_df, response_window=1):
    # Let's start by locating target trials
    task_types = ['oddball', 'oneBack', 'twoBack']
    response_times = response_times[:]  # copy
    target_trial_idx = events_df['trial_type'].isin(task_types)
    nontarget_trial_idx = ~target_trial_idx

    events_df['response_time'] = 'n/a'
    events_df['accuracy'] = 1  # no-response is default, so correct is too
    events_df['classification'] = 'true_negative'
    events_df.loc[target_trial_idx, 'accuracy'] = 0  # default to miss
    events_df.loc[target_trial_idx, 'classification'] = 'false_negative'
    print(response_times)

    # Log hits
    for trial_idx in events_df.index[target_trial_idx]:
        onset = events_df.loc[trial_idx, 'onset']
        keep_idx = []
        # Looping backwards lets us keep earliest response for RT
        for i_resp, rt in enumerate(response_times[::-1]):
            if onset <= rt <= (onset + response_window):
                events_df.loc[trial_idx, 'accuracy'] = 1
                events_df.loc[trial_idx, 'response_time'] = rt - onset
                events_df.loc[trial_idx, 'classification'] = 'true_positive'
            else:
                keep_idx.append(response_times.index(rt))
        response_times = [response_times[i] for i in sorted(keep_idx)]
    print('after logging hits')
    print(response_times)

    # Log false alarms
    for trial_idx in events_df.index[nontarget_trial_idx]:
        onset = events_df.loc[trial_idx, 'onset']
        if trial_idx == events_df.index.values[-1]:
            next_onset = onset + response_window
        else:
            next_onset = events_df.loc[trial_idx+1, 'onset']
        # Looping backwards lets us keep earliest response for RT
        for i_resp, rt in enumerate(response_times[::-1]):
            if onset <= rt < next_onset:
                # Ignore response window and use current trial's duration only
                events_df.loc[trial_idx, 'accuracy'] = 0
                events_df.loc[trial_idx, 'classification'] = 'false_positive'
                events_df.loc[trial_idx, 'response_time'] = rt - onset
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
            lst[1:] = np.random.choice(lst[1:], size=len(lst)-1, replace=False)
        else:
            lst = elems[:]
            np.random.shuffle(lst)
        res.extend(lst)
    return res


def close_on_esc(win):
    """
    Closes window if escape is pressed
    """
    if 'escape' in event.getKeys():
        win.close()
        core.quit()


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
    window.callOnFlip(response.clock.reset)
    event.clearEvents(eventType='keyboard')
    while time.time() - start_time < duration:
        if isinstance(stim, list):
            for s in stim:
                s.draw()
        else:
            stim.draw()
        keys = event.getKeys(keyList=['1', '2'], timeStamped=clock)
        if keys:
            response.keys.extend(keys)
            response.rt.append(response.clock.getTime())
        close_on_esc(win)
        win.flip()
    response.status = STOPPED
    return response.keys, response.rt


if __name__ == '__main__':
    # Ensure that relative paths start from the same directory as this script
    try:
        script_dir = op.dirname(op.abspath(__file__)).decode(sys.getfilesystemencoding())
    except AttributeError:
        script_dir = op.dirname(op.abspath(__file__))

    # Collect user input
    # ------------------
    # Remember to turn fullscr to True for the real deal.
    exp_info = {'Subject': '',
                'Session': '',
                'Task': ['Oddball', 'OneBack', 'TwoBack'],
                'Image Set': ['default', 'alternate', 'both'],
                'Number of Runs': 4,
                'BioPac': ['No', 'Yes']}
    dlg = gui.DlgFromDict(
        exp_info,
        title='Functional localizer: {}'.format(exp_info['Task']),
        order=['Subject', 'Session', 'Task', 'Image Set', 'Number of Runs', 'BioPac'])
    window = visual.Window(
        fullscr=False,
        size=(800, 600),
        monitor='testMonitor',
        units='norm',
        allowStencil=False,
        allowGUI=False,
        color='gray',
        colorSpace='rgb',
        blendMode='avg',
        useFBO=True)
    if not dlg.OK:
        core.quit()

    # Establish serial port connection
    if exp_info['BioPac'] == 'Yes':
        ser = serial.Serial('COM2', 115200)

    # Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    base_name = 'sub-{0}_ses-{1}_task-localizer{2}'.format(
        exp_info['Subject'].zfill(2), exp_info['Session'].zfill(2),
        exp_info['Task'])

    # save a log file for detail verbose info
    filename = op.join(script_dir, 'data/{0}_events'.format(base_name))
    logfile = logging.LogFile(filename+'.log', level=logging.EXP)
    logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

    # Initialize stimuli
    # ------------------
    countdown_text_box = visual.TextStim(
        win=window,
        name='countdown',
        text=None,
        font=u'Arial',
        height=0.1,
        pos=(0, 0),
        wrapWidth=None,
        ori=0,
        color='white',
        colorSpace='rgb',
        opacity=1,
        depth=-1.0)
    if exp_info['Task'] == 'Oddball':
        instruction_text = 'Fixate. Press a button when a scrambled image appears.'
    elif exp_info['Task'] == 'TwoBack':
        instruction_text = 'Fixate. Press a button when an image repeats with one intervening image.'
    else:
        instruction_text = 'Fixate. Press a button when an image repeats on sequential trials.'
    instruction_text_box = visual.TextStim(
        win=window,
        name='instructions',
        text=instruction_text,
        font=u'Arial',
        height=0.1,
        pos=(0, 0),
        wrapWidth=None,
        ori=0,
        color='white',
        colorSpace='rgb',
        opacity=1,
        depth=-1.0)
    stim_image = visual.ImageStim(
        win=window,
        name='stimulus',
        image=None,
        ori=0,
        pos=(0, 0),
        color=[1, 1, 1],
        colorSpace='rgb',
        opacity=1,
        depth=-1.0,
        interpolate=False)
    crosshair = visual.TextStim(
        win=window,
        name='fixation',
        text='+',
        font=u'Arial',
        pos=(0, 0),
        height=0.14,
        wrapWidth=None,
        ori=0,
        color='red',
        colorSpace='rgb',
        opacity=1,
        depth=0.0)
    performance_screen = visual.TextStim(
        win=window,
        name='performance_screen',
        text=None,
        font=u'Arial',
        pos=(0, 0),
        height=0.14,
        wrapWidth=None,
        ori=0,
        color='white',
        colorSpace='rgb',
        opacity=1,
        depth=0.0)

    # Collect stimulus sets
    n_runs = int(exp_info['Number of Runs'])
    if exp_info['Image Set'] == 'default':
        stimulus_folders = {
            'bodies': ['body'],
            'characters': ['word'],
            'faces': ['adult'],
            'objects': ['car'],
            'places': ['house'],
        }
    elif exp_info['Image Set'] == 'alternate':
        stimulus_folders = {
            'bodies': ['limb'],
            'characters': ['number'],
            'faces': ['child'],
            'objects': ['instrument'],
            'places': ['corridor'],
        }
    elif exp_info['Image Set'] == 'both':
        stimulus_folders = {
            'bodies': ['body', 'limb'],
            'characters': ['word', 'number'],
            'faces': ['adult', 'child'],
            'objects': ['car', 'instrument'],
            'places': ['house', 'corridor'],
        }

    n_blocks_per_condition = int(np.floor(N_BLOCKS / len(stimulus_folders.keys())))

    stimuli = {}
    for category in stimulus_folders.keys():
        stimulus_files = [glob(op.join(script_dir, 'stimuli/{}/*.jpg'.format(stimulus_folder))) for
                          stimulus_folder in stimulus_folders[category]]
        stimulus_files = [item for sublist in stimulus_files for item in sublist]
        stimuli[category] = stimulus_files
    scrambled_stimuli = glob(op.join(script_dir, 'stimuli/scrambled/*.jpg'))

    # Scanner runtime
    # ---------------
    global_clock = core.Clock()  # to track the time since experiment started
    run_clock = core.Clock()  # to track time since each run starts (post scanner pulse)
    miniblock_clock = core.Clock()  # to track duration of each miniblock
    trial_clock = core.Clock()  # to track duration of each trial

    for i_run in range(n_runs):
        COLUMNS = ['onset', 'duration', 'trial_type', 'miniblock_number',
                   'category', 'subcategory', 'stim_file']
        run_data = {c: [] for c in COLUMNS}
        run_label = i_run + 1
        outfile = op.join(script_dir, 'data',
                          '{0}_run-{1:02d}_events.tsv'.format(base_name, run_label))

        miniblock_categories = randomize_carefully(list(stimuli.keys()),
                                                   n_blocks_per_condition)

        # Determine which trials will be task
        # This might be overly convoluted, but it maximizes balance between
        # task/non-task instead of just sampling with set probabilities
        nontask_rate = 1 - TASK_RATE
        task_mult = 1 / np.minimum(TASK_RATE, nontask_rate)
        n_task = int(task_mult * TASK_RATE)
        n_nontask = int(task_mult * nontask_rate)
        grabber_list = [1] * n_task + [0] * n_nontask
        n_dupes = int(np.ceil(N_BLOCKS / len(grabber_list)))
        task_miniblocks = grabber_list * n_dupes
        np.random.shuffle(task_miniblocks)

        # Scanner runtime
        # ---------------
        # Wait for trigger from scanner.
        if i_run == 0:
            instruction_text_box.draw()
        else:
            hit_count = (run_frame['classification'] == 'true_positive').sum()
            n_probes = run_frame['classification'].isin(['false_negative', 'true_positive']).sum()
            hit_rate = hit_count / n_probes
            fa_count = (run_frame['classification'] == 'false_positive').sum()
            performance_str = ('Hits: {0}/{1} ({2:.02f}%)\nFalse alarms: {3}').format(
                hit_count, n_probes, hit_rate, fa_count)
            performance_screen.setText(performance_str)
            performance_screen.draw()
        window.flip()
        event.waitKeys(keyList=['5'])

        # Start recording
        if exp_info['BioPac'] == 'Yes':
            ser.write('FF')

        run_clock.reset()
        all_responses, all_response_times = [], []
        countdown_sec = COUNTDOWN_DURATION
        remaining_time = COUNTDOWN_DURATION
        countdown_text_box.setText(countdown_sec)
        while remaining_time > 0:
            countdown_text_box.draw()
            window.flip()
            remaining_time = COUNTDOWN_DURATION - run_clock.getTime()
            if np.floor(remaining_time) <= countdown_sec:
                countdown_text_box.setText(countdown_sec + 1)
                countdown_sec -= 1

        real_countdown_duration = run_clock.getTime()
        run_data['onset'].append(0)
        run_data['duration'].append(real_countdown_duration)
        run_data['trial_type'].append('countdown')
        run_data['stim_file'].append('n/a')
        run_data['category'].append('n/a')
        run_data['subcategory'].append('n/a')
        run_data['miniblock_number'].append('n/a')

        for j_miniblock, category in enumerate(miniblock_categories):
            miniblock_clock.reset()
            # Block of stimuli
            miniblock_stimuli = list(np.random.choice(stimuli[category],
                                                      size=N_STIMULI_PER_BLOCK,
                                                      replace=False))
            if task_miniblocks[j_miniblock] == 1:
                # Adjust stimuli based on task
                if exp_info['Task'] == 'Oddball':
                    target_idx = np.random.choice(len(miniblock_stimuli))
                    scrambled_stim = np.random.choice(scrambled_stimuli)
                    miniblock_stimuli[target_idx] = scrambled_stim
                elif exp_info['Task'] == 'OneBack':
                    # target is second stim of same kind
                    target_idx = np.random.choice(len(miniblock_stimuli) - 1) + 1
                    miniblock_stimuli[target_idx] = miniblock_stimuli[target_idx - 1]
                elif exp_info['Task'] == 'TwoBack':
                    # target is second stim of same kind
                    target_idx = np.random.choice(len(miniblock_stimuli) - 2) + 2
                    miniblock_stimuli[target_idx] = miniblock_stimuli[target_idx - 2]
            else:
                target_idx = None

            for k_stim, stim_file in enumerate(miniblock_stimuli):
                trial_clock.reset()
                onset_time = run_clock.getTime()
                stim_image.image = stim_file
                task_responses, _ = draw(win=window, stim=stim_image,
                                         duration=IMAGE_DURATION,
                                         clock=run_clock)
                all_responses += [resp[0] for resp in task_responses]
                all_response_times += [resp[1] for resp in task_responses]
                stim_image.size = None  # clear size
                duration = trial_clock.getTime()
                isi_dur = np.maximum((IMAGE_DURATION + TARGET_ISI) - duration, 0)
                rest_responses, _ = draw(win=window, stim=crosshair,
                                         duration=isi_dur, clock=run_clock)
                all_responses += [resp[0] for resp in rest_responses]
                all_response_times += [resp[1] for resp in rest_responses]
                relative_stim_file = op.sep.join(stim_file.split(op.sep)[-2:])
                subcategory = stim_file.split(op.sep)[-2]

                if k_stim == target_idx:
                    trial_type = exp_info['Task'].lower()
                else:
                    trial_type = 'baseline'

                # Log info
                run_data['onset'].append(onset_time)
                run_data['duration'].append(duration)
                run_data['trial_type'].append(trial_type)
                run_data['stim_file'].append(relative_stim_file)
                run_data['category'].append(category)
                run_data['subcategory'].append(subcategory)
                run_data['miniblock_number'].append(j_miniblock + 1)
            miniblock_duration = miniblock_clock.getTime()

        run_frame = pd.DataFrame(run_data)
        run_frame.to_csv(outfile, sep='\t', line_terminator='\n', na_rep='n/a',
                         index=False, float_format='%.2f')
        run_frame = allocate_responses(all_responses, all_response_times, run_frame,
                                       response_window=RESPONSE_WINDOW)
        run_frame.to_csv(outfile, sep='\t', line_terminator='\n', na_rep='n/a',
                         index=False, float_format='%.2f')

        # Last fixation
        last_iti = TOTAL_DURATION - run_clock.getTime()
        draw(win=window, stim=crosshair, duration=last_iti, clock=run_clock)

        # End recording
        if exp_info['BioPac'] == 'Yes':
            ser.write('00')

        print(all_responses)
        print(all_response_times)
        print('Total duration of run: {}'.format(run_clock.getTime()))
    # end run_loop

    # Shut down serial port connection
    if exp_info['BioPac'] == 'Yes':
        ser.close()

    # Scanner is off for this
    hit_count = (run_frame['classification'] == 'true_positive').sum()
    n_probes = run_frame['classification'].isin(['false_negative', 'true_positive']).sum()
    hit_rate = hit_count / n_probes
    fa_count = (run_frame['classification'] == 'false_positive').sum()
    performance_str = ('Hits: {0}/{1} ({2:.02f}%)\nFalse alarms: {3}').format(
        hit_count, n_probes, hit_rate, fa_count)
    performance_screen.setText(performance_str)
    draw(win=window, stim=performance_screen, duration=END_SCREEN_DURATION, clock=global_clock)
    window.flip()

    logging.flush()

    # make sure everything is closed down
    window.close()
    core.quit()
