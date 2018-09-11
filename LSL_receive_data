"""
Created on Tue Sep 11 09:13:53 2018

@author: gretatuckute

Reads multi-channel time series from EEG (Enobio needs to be connected via wifi)

This version runs without activating LSL LabRecorder.exe (NIC2.0 running - with LSL outlet)
"""

# Imports 
from pylsl import StreamInlet, resolve_stream, local_clock

streams = resolve_stream('type', 'EEG') # Searches for EEG stream

print("Number of streams found: %s " % len(streams))

# Create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

sample, timestamp = inlet.pull_sample()

print(sample)
print(len(sample))
print(timestamp)

# Can also read time series from LSL in chunks
# chunk, timestamps = inlet.pull_chunk()


#while True:
    # get a new sample 
#    sample, timestamp = inlet.pull_sample()
#print(timestamp, sample)

# If I want to check the most recent sample of a stream, try pull_sample()

# Implement local_clock()

# LSL time stamps the data by push_sample() or to call the local_clock() function to read out the LSL clock, and then pass that in
