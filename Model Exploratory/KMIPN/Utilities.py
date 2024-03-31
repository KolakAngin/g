import pandas as pd
import numpy as np

from tqdm.auto import tqdm

def insertDataKetinggalan(dataFrame : pd.DataFrame,
                          pasalBefore : list[str],
                          namePasal : list[str],
                          pasalChange : list[str]) -> pd.DataFrame :
    
    for before,name,pasal in zip(pasalBefore,namePasal,pasalChange):
        indexBefore = dataFrame[dataFrame["Pasal"] ==  before].index[0] + 0.5
        dataFrame.loc[indexBefore] = [name,pasal]
        dataFrame = dataFrame.sort_index().reset_index(drop=True)
    return dataFrame

def lihatPasal(dataFrame : pd.DataFrame, pasal : str | int) -> pd.DataFrame:
    return dataFrame[dataFrame["Pasal"] == f"Pasal {pasal}"]

def insertBabToData(dataFrame : pd.DataFrame, rangeBab : list[str], nameBab : str, returnDf : bool = False) -> None | pd.DataFrame:
    rangeBab = [f"Pasal {a}" for a in rangeBab]

    startLocation = dataFrame[dataFrame["Pasal"] == rangeBab[0]].index[0]
    if len(rangeBab) == 1:
        endLocation = dataFrame[dataFrame["Pasal"] == rangeBab[0]].index[0]
    else:
        endLocation = dataFrame[dataFrame["Pasal"] == rangeBab[1]].index[0]
    
    # memasukan data bab
    dataFrame.loc[startLocation:endLocation,"Bab"] = nameBab
    dataFrame["Bab"].astype("str")

    if(returnDf):
        return dataFrame
     


def insertBagianToData(dataFrame : pd.DataFrame, rangeBab : list[str], nameBab : str, returnDf : bool = False) -> None | pd.DataFrame:
    rangeBab = [f"Pasal {a}" for a in rangeBab]
    startLocation = dataFrame[dataFrame["Pasal"] == rangeBab[0]].index[0]
    if len(rangeBab) == 1:
        endLocation = dataFrame[dataFrame["Pasal"] == rangeBab[0]].index[0]
    else:
        endLocation = dataFrame[dataFrame["Pasal"] == rangeBab[1]].index[0]
    
    # memasukan data bagian
    dataFrame.loc[startLocation:endLocation,"Bagian"] = nameBab
    dataFrame["Bagian"].astype("str")

    if(returnDf):
        return dataFrame
    


def iterationInsert(status : str, dfTarget : pd.DataFrame, dfChange : pd.DataFrame,func) -> None | list:
    title =  dfChange[status.title()].tolist()
    rangePasal = [a.split("-") for a in dfChange["Range Pasal"]]
    listOfError = []
    for t,r in tqdm(zip(title,rangePasal),desc=f"Inserting {status}"):
        try :
            func(dfTarget,r,t)
        except :
            print(f"{bcolors.BOLD}{bcolors.FAIL}ada kesalahan di {r}{bcolors.ENDC}")
            for i in r:
                if (len(lihatPasal(dfTarget,i)) == 0):
                    print(f"{bcolors.BOLD}{bcolors.FAIL}indikasi ada kesalahan di Pasal {i}{bcolors.ENDC}")
                    listOfError.append(i)
    
    if len(listOfError) > 1 :
        return listOfError


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
