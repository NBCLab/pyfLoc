import psychopy
from psychopy import visual, core

N_STIMULI_PER_BLOCK = 12
IMAGE_DURATION = 0.5
TOTAL_DURATION = 240  # four minutes


if __name__ == '__main__':
    # Collect user input
    # ------------------
    # Remember to turn fullscr to True for the real deal.
    exp_info = {'subject': '',
                'session': '',
                'task': ['oddball', 'oneBack', 'twoBack'],
                'image set': ['default', 'alternate', 'both'],
                'number of runs': 4}
    dlg = psychopy.gui.DlgFromDict(
        exp_info,
        title='Functional localizer: {}'.format(exp_info['task']),
        order=['subject', 'session'])
    window = psychopy.visual.Window(
        size=(800, 600), fullscr=True, monitor='testMonitor', units='deg',
        # size=(500, 400), fullscr=False, monitor='testMonitor', units='deg',
        allowStencil=False, allowGUI=False, color='black')
    stim_image = visual.ImageStim(
        win=window, name='stimulus', image=None, size=(0.25, 0.375),
        ori=0, pos=(0, 0), color=[1, 1, 1], colorSpace='rgb', opacity=1,
        depth=-1.0, interpolate=True)
    crosshair = psychopy.visual.TextStim(
        window, '+', height=2, name='crosshair', color='white')

    if not dlg.OK:
        psychopy.core.quit()

    n_runs = int(exp_info['number of runs'])
    if exp_info['image set'] == 'default':
        stimulus_folders = {
            'bodies': ['body'],
            'characters': ['word'],
            'faces': ['adult'],
            'objects': ['car'],
            'places': ['house'],
        }
    elif exp_info['image set'] == 'alternate':
        stimulus_folders = {
            'bodies': ['limb'],
            'characters': ['number'],
            'faces': ['child'],
            'objects': ['instrument'],
            'places': ['corridor'],
        }
    elif exp_info['image set'] == 'both':
        stimulus_folders = {
            'bodies': ['body', 'limb'],
            'characters': ['word', 'number'],
            'faces': ['adult', 'child'],
            'objects': ['car', 'instrument'],
            'places': ['house', 'corridor'],
        }

    stimuli = {}
    for stimtype in stimulus_folders.keys():
        stimulus_files = [glob('stimuli/{}/*.jpg'.format(stimulus_folder)) for
                          stimulus_folder in stimulus_folders[stimtype]]
        stimulus_files = [item for sublist in stimulus_files for item in sublist]
        stimuli[stimtype] = stimulus_files

    for i_run in range(1, n_runs+1):
        filename = ('data/sub-{0}_ses-{1}_task-localizer{2}'
                    '_run-{3}_events').format(exp_info['subject'],
                                              exp_info['session'],
                                              exp_info['task'].title(),
                                              i_run)
        df = pd.DataFrame(columns=['onset', 'duration', 'category', 'stimulus'])



        # Block of stimuli
        miniblock_stimuli = random.sample(stimuli[stimtype], N_STIMULI_PER_BLOCK)
        for stim in miniblock_stimuli:
            stim_image.image = stim
            stim_image.autoDraw = True  # Automatically draw every frame
            onset_time = global_clock.getTime()
            window.flip()
            core.wait(IMAGE_DURATION)
            relative_stim_file = op.sep.join(stim.split(op.sep)[-2:])
            category = stim.split(op.sep)[-2]
            row = {
                'stimulus': relative_stim_file,
                'category': category,
                'onset': onset_time,
                'duration': IMAGE_DURATION,
            }
            df.append(row)
