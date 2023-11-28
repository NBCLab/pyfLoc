# pyfLoc

A PsychoPy implementation of the Stanford VPN lab's functional localizer task.

This implementation is not associated with the VPN lab, so any bugs are our own.
The Python code for this task is released under our selected license,
but the stimuli for the task must be taken from the VPN lab's [fLoc repository](https://github.com/VPNL/fLoc),
and any use of the stimuli should include an acknowledgement in any resulting research products
(see [here](https://github.com/VPNL/fLoc#citation)).

Please download the stimuli from that repository,
then place them in a folder named "stimuli" in the same folder as `fLoc.py` and `config.yml`.
Alternatively, you can use your own stimuli,
as long as you use the appropriate folder structure and update the config file.

## Implementation information

This functional localizer task uses a miniblock design with different categories of images.
To keep participants focused on the task,
there is also a behavioral task that participants will perform throughout the localizer.
There are three options for this task: oddball, one-back, and two-back.

The task does not include any kind of jitter.

The task is set up for acquisition in an MRI,
so each run starts when a scanner pulse is sent (i.e., the "5" key).

Any button press using the buttons "1", "2", "3", or "4" counts as a response.

### Configuration

You should modify the configuration file (`config.yml`) based on the following elements:

1.  The current configuration has 4-minute runs. The VPN lab recommends acquiring at least 4 runs.
2.  The repetition time (TR) of fMRI data for the localizer experiment must be a factor of its miniblock duration (6 s by default).
    The original paper used a TR of 2 seconds.
3.  If you want, you can replace the VPN lab's stimuli with another set.
    Each stimulus set *must* have a "baseline" category (for all tasks) and a "scrambled" category (for the oddball task).

### Differences from the original task

While we have attempted to maximize similarity between this task and the VPN lab's task,
there are differences.
I'm not sure what those differences are, unfortunately, since I wrote this task a few years ago.
