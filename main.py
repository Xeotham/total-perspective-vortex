import numpy as np
import mne
import pathlib as pth
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Qt5Agg')
data_folder = pth.Path('eeg-motor-movementimagery-dataset-1.0.0/files')

# Change with a iterdir loop or a data selector.
s001 = data_folder / 'S001'

s001r01 = s001 / 'S001R01.edf'

raw = mne.io.read_raw_edf(s001r01)



raw.plot()
raw.compute_psd().plot()
plt.show()