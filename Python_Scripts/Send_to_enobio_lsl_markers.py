####		Send to Enobio LSL markers 		####

# An empty vector is created to hold the marker.
# After a random delay that can last up to 3 seconds the marker is sent to enobio.


class data:
    def __init__(self, fs,filename=None):
        self.fs,self.filename = fs,filename
glo=data(500)
#glo.options.filename=time.strftime("%H%M%S_%d%m%Y")
def save_data(glo,eeg,timestamps):
    #if exist(glo.options.filename)
    if glo.filename==None:
        glo.filename='data_send_'+time.strftime("%H%M%S_%d%m%Y")+'.csv'
        with open(glo.filename,'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['time stamp','marker'])
    with open(glo.filename,'a') as csvfile:
        #fieldnames=['name1','name2']
        writer = csv.writer(csvfile)
        #writer.writeheader()
        writer.writerow([eeg,timestamps])
    return glo

#%%
from pylsl import StreamInfo, StreamOutlet
import time
import random
import numpy as np

print ("Creating a new marker stream info...\n")
info = StreamInfo('MyMarkerStream3','Markers',1,0,'int32','myuniquesourceid23443')

print("Opening an outlet...\n")
outlet =StreamOutlet(info)

print("Sending data...\n")
mkr=0
c=0
TIME=np.zeros(1000)
vec = []
while (True):
    time.sleep(0.5) #random.randint(0,3)
    mkr = mkr+1
    vec.append(mkr)
    
    TIME[c]=time.time()
    outlet.push_sample(vec) 
    print("Now sending: \t" + str(vec)+"\n")
    c=c+1

#%%
from tempfile import TemporaryFile
send_marker_file = TemporaryFile()
x = np.arange(10)
np.save(send_marker_file, TIME)
send_marker_file.seek(0)
np.load(send_marker_file)



