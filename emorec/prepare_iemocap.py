"""
Created on Tue May 23 17:24:02 2017

@author: eesungkim

Src repo: https://github.com/eesungkim/Speech_Emotion_Recognition_DNN-ELM

Modified on June 2019 by Santi Pascual

"""
import os
import re
import scipy.io.wavfile
import numpy as np
from os import listdir
from os.path import isfile, join

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def load_utterInfo(inputFile):
    pattern = re.compile('[\[]*[0-9]*[.][0-9]*[ -]*[0-9]*[.][0-9]*[\]][\t][a-z0-9_]*[\t][a-z]{3}[\t][\[][0-9]*[.][0-9]*[, ]+[0-9]*[.][0-9]*[, ]+[0-9]*[.][0-9]*[\]]',re.IGNORECASE)
    with open (inputFile, "r") as myfile:
        data=myfile.read().replace('\n', ' ')
    result = pattern.findall(data)
    out = []
    for i in result:
        a = i.replace('[','')
        b = a.replace(' - ','\t')
        c = b.replace(']','')
        x = c.replace(', ','\t')
        out.append(x.split('\t'))
    return out

def make5thWaves(pathSession):
    txtPattern = re.compile('[.]+(txt)$',re.IGNORECASE)
    pathEmo = pathSession+'/dialog/EmoEvaluation/'
    pathWav = pathSession+'/dialog/wav/'
    pathWavFolder1 = pathSession+'/sentences/wav1/'
    for emoFile in [f for f in listdir(pathEmo) if isfile(join(pathEmo, f))]:
        path=pathWav+txtPattern.split(emoFile)[0]+'.wav'
        (sr,signal) = scipy.io.wavfile.read(path,mmap=False)
        for utterance in (load_utterInfo(pathEmo+emoFile)):
                t0 = int(np.ceil(float(utterance[0])*sr))
                tf = int(np.ceil(float(utterance[1])*sr))
                
                if(utterance[2][-4]=='F'): # Session 1만 L(0) channel : main, R(0): 보조
                        mono = signal[t0:tf][:,0]
                else:   
                        mono = signal[t0:tf][:,1]
                
                folderpath=pathWavFolder1+utterance[2][:-5]
                makedirs(folderpath)
                fileName=pathWavFolder1+utterance[2][:-5]+'/'+utterance[2]+'.wav'
                scipy.io.wavfile.write(fileName,16000, mono)
                
def load_session(pathSession):
    pathEmo = pathSession+'/dialog/EmoEvaluation/'
    pathWavFolder = pathSession+'/sentences/wav/'
    
    improvisedUtteranceList = []
    for emoFile in [f for f in listdir(pathEmo) if isfile(join(pathEmo, f))]:
        for utterance in (load_utterInfo(pathEmo+emoFile)):
            # if ((utterance[3] == 'neu') or (utterance[3] == 'hap') or (utterance[3] == 'sad') or (utterance[3] == 'fru') or (utterance[3] =='exc')):
            if ((utterance[3] == 'neu') or (utterance[3] == 'hap') or (utterance[3] == 'sad') or (utterance[3] == 'ang') or (utterance[3] =='exc')):
                path=pathWavFolder+utterance[2][:-5]+'/'+utterance[2]+'.wav'
                (sr,signal) = scipy.io.wavfile.read(path,mmap=False)
        
                if(emoFile[7] != 'i'):
                    if(utterance[2][7] =='s'):
                        improvisedUtteranceList.append([signal,utterance[3],utterance[2][18]])
                    else:
                        improvisedUtteranceList.append([signal,utterance[3],utterance[2][15]])
                else:
                    improvisedUtteranceList.append([signal,utterance[3],utterance[2][15]])
    return improvisedUtteranceList

def count_emotion(session):
    dic={'neu':0, 'hap':0, 'sad':0, 'ang':0, 'sur':0, 'fea':0, 'dis':0, 'fru':0, 'exc':0, 'xxx':0}
    for i in range(len(session)):
        if(session[i][1] == 'neu'): dic['neu']+=1
        elif(session[i][1] == 'hap'): dic['hap']+=1
        elif(session[i][1] == 'sad'): dic['sad']+=1
        elif(session[i][1] == 'ang'): dic['ang']+=1
        elif(session[i][1] == 'sur'): dic['sur']+=1
        elif(session[i][1] == 'fea'): dic['fea']+=1
        elif(session[i][1] == 'dis'): dic['dis']+=1
        elif(session[i][1] == 'fru'): dic['fru']+=1
        elif(session[i][1] == 'exc'): dic['exc']+=1
        elif(session[i][1] == 'xxx'): dic['xxx']+=1
    return dic

# Save classified wave files to manipulate easily
def save_wavFile(session, pathName):
    makedirs(pathName)
    for idx,utterance in enumerate(session):
        label = utterance[1]
        if(label=='exc'):
            # label='exc'
            label='hap'
        directory = "%s/%s" % (pathName,label)
        makedirs(directory)
        filename = "%s/psn%s%s_%s_s%03d_orgn.wav" % (directory,pathName[-4],pathName[-2],label,idx)      
        scipy.io.wavfile.write(filename,16000, utterance[0])

    return 0

def main05(ROOT_PATH, path_loadSession, path_directory,MODEL_NAME):
    #make5thWaves("%s%s"%(path_loadSession,5))
    for k in range(5):
        session_=[]
        session = load_session("%s%s"%(path_loadSession,k+1))
        for idx in range(len(session)):
            session_.append(session[idx])

        dic_ = count_emotion(session_)
        print('='*50)
        print('Total Session_%d :'%(k+1) +" %d"%sum(dic_.values()))
        print(dic_)
        pathName1 = "%s/session%d/" %(path_directory,(k+1) )
        print('='*50)
        if save_wavFile(session_,pathName1) == 0 :
            print('Completed to save session_%d Wave files successfully.' %(k+1))
    print('='*50)

if __name__ == '__main__':
    ROOT_PATH = os.getcwd()
    AUDIO_DIR = 'data/IEMOCAP_full_release/Session'
    MOD_AUDIO_DIR = 'data/IEMOCAP_ahsn_leave-two-speaker-out'
    LOG_NAME='IEMOCAP'
    ORIGINAL_DATASET_PATH = os.path.join(ROOT_PATH, AUDIO_DIR)
    MODIFIED_DATASET_PATH = os.path.join(ROOT_PATH, MOD_AUDIO_DIR)
    main05(ROOT_PATH,ORIGINAL_DATASET_PATH,MODIFIED_DATASET_PATH,LOG_NAME)

