# Imports
from pylsl import StreamInlet, resolve_stream, local_clock
import numpy as np
import matplotlib.pyplot as plt
import time

streams = resolve_stream('type', 'EEG')  # Searches for EEG stream
inlet = StreamInlet(streams[0])
sample, timestamp = inlet.pull_sample()

plt.ion()
fig = plt.figure()
axes = fig.add_subplot(111)

axes.set_autoscale_on(True)
axes.autoscale_view(True, True, True)

l, = plt.plot([], [], 'r-')
plt.xlabel('timestamp')
plt.ylabel('EEG channel 1 sample')
plt.title('EEG test')

xdata = [timestamp]
ydata = [sample[0]]

# while True:
for i in range(0, 1000):
    # eeg_data, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=32)
    sample, timestamp = inlet.pull_sample()
    #    eeg = np.array(eeg_data)
    #    ch1 = eeg[:,1]
    ch1 = sample[0]
    # time.sleep(0.1)
    ydata.append(ch1)
    xdata.append(timestamp)
    l.set_data(xdata, ydata)
    # print(xdata) # Prints the precise time stamp used for x axis
    # print(ydata) # Prints the ch1 sample from EEG stream
    axes.relim()
    axes.autoscale_view(True, True, True)
    plt.draw()

