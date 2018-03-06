import numpy as np
import pandas as pd
PATH1 = "ABIq.csv"
ABI_df = pd.read_csv(PATH1, header=0,sep="|")   #

agencyList = ABI_df.Agency.unique()
SKU_List = ABI_df.SKU.unique()
Mon_List = list(sorted(ABI_df.YearMonth.unique())) #sorted




def crosstabs(factorlist1,factorlist2,factorname1,factorname2,field):
    global inuse
    output = np.zeros(( len(factorlist1),len(factorlist2) ))
    
    for k, eachS in enumerate(factorlist2):
        for j,eachA in enumerate(factorlist1):
            output[j,k] =ABI_df.loc[(ABI_df[factorname1] == eachA) & (ABI_df[factorname2] == eachS) ].Volume.mean()
            #print (eachA,eachS, output[j,k])
    
    inuse = ~np.isnan(output) #index of relevant pairs
    return output


#average monthly volume per AS [Agency-SKU]
#later we normalize all months to ending run rate for growth, seasonality, etc.

avgvolAS = crosstabs(agencyList, SKU_List, "Agency", "SKU","Volume")
inuseAS = inuse
# np.sum(inuseAS)  350 vs 58x25


#average sales by agency per month across all products
avgvolAD = crosstabs(agencyList, Mon_List, "Agency", "YearMonth","Volume")
inuseAM = inuse #3441 of 3480



def count2way(factorlist1,factorlist2,factorname1,factorname2):
    output = np.zeros(( len(factorlist1),len(factorlist2) ))
    for k, eachS in enumerate(factorlist2):
        for j,eachA in enumerate(factorlist1):
            output[j,k] =ABI_df.loc[(ABI_df[factorname1] == eachA) & (ABI_df[factorname2] == eachS) ].Volume.count()
    return output


countAS = count2way(agencyList, SKU_List, "Agency", "SKU") #needed
#252 of the 350 pairs have 60 


#vol index
#1)
#2 our volume we can build a volume index based on markets we participaed all 60 periods. this might be obscured by our mkt share trend etc.

ABIvol = np.zeros(60)
for k, eachS in enumerate(SKU_List):
    for j,eachA in enumerate(agencyList):
        if countAS[j,k]==60:
            for l,each3 in enumerate(Mon_List):
                ABIvol[l] += ABI_df.loc[(ABI_df["Agency"] == eachA) & (ABI_df["SKU"] == eachS) & (ABI_df["YearMonth"] == each3) ].Volume.iloc[0]
    
INDYvol = np.zeros(60)
for l,each3 in enumerate(Mon_List):
     INDYvol[l] =ABI_df.loc[(ABI_df["Agency"] == "Agency_38") & (ABI_df["SKU"] == "SKU_07") & (ABI_df["YearMonth"] == each3) ].indy_Volume.iloc[0]
     #print(l,each3,INDYvol[l])
     #this agency SKU pair is 

def normalz(vector):
    return vector/vector.mean()

ABIvolN = normalz(ABIvol)
INDYvolN = normalz(INDYvol )


#regress ABIvolN against month(0 to 59) 12 dummmy variables and constant. 
#from Seasonality1.xls

SeasL = [0.778283664, 0.863838245, 1.030756782, 1.078964957, 1.079718092, 1.023899113,
                  1.010547092, 1.026186344, 0.957600067,0.938439032, 0.869697851, 1.095641188]
#note this averages to .979 because excludes vol growth


def MonthIdx(YYYYMM): 
     return Mon_List.index(YYYYMM) 
def MI_from_MonthIdx(i): 
     return i % 12
 
def MonthSeas(MIvalue): 
    return SeasL[MIvalue]

ABI_df["MQ"] =ABI_df["YearMonth"].apply(MonthIdx) #0 to 59
ABI_df["MI"] =ABI_df["MQ"].apply(MI_from_MonthIdx) #0 to 59
ABI_df["YY"] = (ABI_df["MQ"] - ABI_df["MI"])/12 +2013
ABI_df["Seas"] =ABI_df["MI"].apply(MonthSeas)
ABI_df["SeasAdjVol"] =ABI_df["Volume"]/ABI_df["Seas"] #SAV

ABI_df.groupby("MI").mean()
#Compares Jan to rest of year , averaging across all AS pairs
#SAV has been averaging around 1730. Jan SAV seems to be a little higher

CY_Summary=ABI_df.groupby("YY").mean()

#not really useful
#avgvolAS_SAV = crosstabs(agencyList, SKU_List, "Agency", "SKU","SeasAdjVol")

import statsmodels.formula.api as sm

#in 3rd dim, first col is slope, second and third are coefs, third is count
#4th is rsqured 567 get pvalues

def AS_Regression(Xfactor): 
    global AScoefs
    AScoefs = np.zeros((58,25,8)) 
    for k, eachS in enumerate(SKU_List):
        for j,eachA in enumerate(agencyList):
            if countAS[j,k]>0:
                AS_subset =  ABI_df.loc[(ABI_df["Agency"] == eachA) & (ABI_df["SKU"] == eachS)]       
                result = sm.ols(formula= "SeasAdjVol ~ MQ + "+Xfactor, data=AS_subset).fit()
                AScoefs[j,k,:3] = result.params
                AScoefs[j,k,3] = AS_subset.shape[0]
                AScoefs[j,k,4] = result.rsquared
                AScoefs[j,k,5:8] = result.pvalues
            else:
                AScoefs[j,k,:] = np.nan
    
    return np.nanmean(AScoefs,axis=(0,1))

diag = 0
if diag==1:  
    predictors = ["Sales2HL","Promotions","Price_HL","Xmas","Max_Temp","Soda_Volume","indy_Volume",
                  "NewYear","Independence_Day","MusicFest","LaborDay","Beer_Capital",
                  "RevolutionDayMem","Easter_Day","GoodFriday","Regional_Games","FIFA_U17_WCup","Football_Gold_Cup"]
    
    for each in predictors:
        print(each, AS_Regression(each),sep="|")
                    
                                       
def AS_2XFactors(Xfactor1,Xfactor2): 
    global AScoefs
    AScoefs = np.zeros((58,25,10)) 
    for k, eachS in enumerate(SKU_List):
        for j,eachA in enumerate(agencyList):
            if countAS[j,k]>0:
                AS_subset =  ABI_df.loc[(ABI_df["Agency"] == eachA) & (ABI_df["SKU"] == eachS)]       
                result = sm.ols(formula= "SeasAdjVol ~ MQ + "+Xfactor1+" + "+Xfactor2, data=AS_subset).fit()
                AScoefs[j,k,:4] = result.params
                AScoefs[j,k,4] = AS_subset.shape[0]
                AScoefs[j,k,5] = result.rsquared
                AScoefs[j,k,6:10] = result.pvalues
            else:
                AScoefs[j,k,:] = np.nan
    
    return np.nanmean(AScoefs,axis=(0,1))

if diag==1:                                        
    predictorZ = ["Sales2HL","Promotions","Price_HL","Xmas","Max_Temp","Soda_Volume","indy_Volume"]
    
    for i,eachA in enumerate(predictorZ[:-1]):
        for eachS in predictorZ[i+1:]:
            print(eachA,eachS, AS_2XFactors(eachA,eachS),sep="|")

        
#avg in January for varies fields including Max temp group by AS
JanEffect = np.zeros((58,25,6)) 
for k, eachS in enumerate(SKU_List):
    for j,eachA in enumerate(agencyList):
        if countAS[j,k]>0:          
            ASQ =  ABI_df.loc[(ABI_df["Agency"] == eachA) & (ABI_df["SKU"] == eachS) & (ABI_df["MI"] ==0)] 
            JanEffect[j,k,:] = np.array([ASQ.Max_Temp.mean(),ASQ.Sales2HL.mean(),
                      ASQ.Promotions.mean(),ASQ.Soda_Volume.mean(),ASQ.indy_Volume.mean(),
                      ASQ.Price_HL.mean()])
        else:
            JanEffect[j,k,:] = np.nan
np.nanmean(JanEffect,axis=(0,1))

#predict Sales2HL for Jan18 using MQ and Seas Even though Seas was developed as a measure of vol fluctuation, 
#it is doing double duty here as a possible explanatory variable for seas price fluctuations
Jan18_Unit_Price_AS = np.zeros((58,25))
Jan13_Unit_Price_AS = np.zeros((58,25))
for k, eachS in enumerate(SKU_List):
    for j,eachA in enumerate(agencyList):
        if countAS[j,k]>0:
            AS_subset =  ABI_df.loc[(ABI_df["Agency"] == eachA) & (ABI_df["SKU"] == eachS)]       
            result = sm.ols(formula= "Sales2HL ~ MQ + Seas", data=AS_subset).fit()
            Jan18_Unit_Price_AS[j,k] = np.dot(result.params,[1,60,SeasL[0]] )
            Jan13_Unit_Price_AS[j,k] = np.dot(result.params,[1,0,SeasL[0]] )
        else:
            Jan18_Unit_Price_AS[j,k] = np.nan

np.nanmean(Jan18_Unit_Price_AS)

#avg in January17 for varies fields including Max temp group by AS
Jan17Effect = np.zeros((58,25,7)) 
for k, eachS in enumerate(SKU_List):
    for j,eachA in enumerate(agencyList):
        if countAS[j,k]>0:          
            ASQ =  ABI_df.loc[(ABI_df["Agency"] == eachA) & (ABI_df["SKU"] == eachS) & (ABI_df["MQ"] ==48)] 
            Jan17Effect[j,k,:] = np.array([ASQ.Max_Temp.mean(),ASQ.Sales2HL.mean(),
                      ASQ.Promotions.mean(),ASQ.Soda_Volume.mean(),ASQ.indy_Volume.mean(),
                      ASQ.Price_HL.mean(),ASQ.SeasAdjVol.mean()])
        else:
            Jan17Effect[j,k,:] = np.nan
np.nanmean(Jan17Effect,axis=(0,1))
    
Jan16Effect = np.zeros((58,25,7)) 
for k, eachS in enumerate(SKU_List):
    for j,eachA in enumerate(agencyList):
        if countAS[j,k]>0:          
            ASQ =  ABI_df.loc[(ABI_df["Agency"] == eachA) & (ABI_df["SKU"] == eachS) & (ABI_df["MQ"] ==36)] 
            Jan16Effect[j,k,:] = np.array([ASQ.Max_Temp.mean(),ASQ.Sales2HL.mean(),
                      ASQ.Promotions.mean(),ASQ.Soda_Volume.mean(),ASQ.indy_Volume.mean(),
                      ASQ.Price_HL.mean(),ASQ.SeasAdjVol.mean()])
        else:
            Jan16Effect[j,k,:] = np.nan
np.nanmean(Jan16Effect,axis=(0,1))    

#study of pairs with short history//informs tje .71 factor below 
for k, eachS in enumerate(SKU_List):
    for j,eachA in enumerate(agencyList):
        if 0<countAS[j,k]<40:
            AS_subset =  ABI_df.loc[(ABI_df["Agency"] == eachA) & (ABI_df["SKU"] == eachS)& (ABI_df["err"] != "datacheck") ]
            AS_subset17=AS_subset.loc[ABI_df["YY"] > 2016.8] ["SeasAdjVol"].mean()
            AS_subset16= AS_subset.loc[(ABI_df["YY"] > 2015.8) & (ABI_df["YY"] < 2016.8) ] ["SeasAdjVol"].mean()
            print( eachA,eachS, AS_subset17,AS_subset16,AS_subset17/AS_subset16-1 )

#Jan temp by Agency
Jan_subdf =  ABI_df.loc[(ABI_df["MI"] == 0)& (ABI_df["err"] != "datacheck") ]            
JanTemp = Jan_subdf.groupby("Agency").Max_Temp.mean()

AS3way = np.zeros((58,25,13)) 

#the 13th col in third dimension is the prediciton for Jan2018
for k, eachS in enumerate(SKU_List):
    for j,eachA in enumerate(agencyList):
        if countAS[j,k]>0:
            AS_subset =  ABI_df.loc[(ABI_df["Agency"] == eachA) & (ABI_df["SKU"] == eachS)& (ABI_df["err"] != "datacheck") ]
            AS_subset17=AS_subset.loc[AS_subset["YY"] > 2016.8]             
            if countAS[j,k]>40:
                result = sm.ols(formula= "SeasAdjVol ~ MQ + Sales2HL + Promotions + Max_Temp", data=AS_subset).fit()
                AS3way[j,k,:5] = result.params
                AS3way[j,k,5] = AS_subset.shape[0]
                AS3way[j,k,6] = result.rsquared
                AS3way[j,k,7:12] = result.pvalues
                promote=AS_subset17["Promotions"].mean()*1.151 
                temperature= JanTemp[eachA]
                #promotions incrd 27% '17 vs '16 and even faster in '16. 7months vs 12 . wont necessarily grow at same rate
                AS3way[j,k,12] =  np.dot(result.params,[1,60,Jan18_Unit_Price_AS[j,k],promote,temperature] ) 
                #if j==3 and k==3:                     print (result.summary())

            else: #for AS pairs with short history, ratio off 2017 
                #on average these markets saw SAV decline 45%. I prorate this for partial yearperhaps some SKUs were abandoned
                AS3way[j,k,:12] = np.nan
                AS3way[j,k,12] = AS_subset17["SeasAdjVol"].mean()  *.71
            
        else:
            AS3way[j,k,:] = np.nan

np.nanmean(AS3way,axis=(0,1))


aLL = list(agencyList)
SKUL = list(SKU_List)


def VolPredict(row): 
    eachA = row["Agency"]
    j=aLL.index(eachA)
    eachS = row["SKU"]
    k=SKUL.index(eachS) 
    period = row["MQ"]
    month = row["MI"]
    AS_subset =  ABI_df.loc[(ABI_df["Agency"] == eachA) & (ABI_df["SKU"] == eachS)& (ABI_df["err"] != "datacheck") ]
    AS_subset17=AS_subset.loc[AS_subset["YY"] > 2016.8]      
    params= AS3way[j,k,:5] 
    prd = (period-53.5)/60
    promote=AS_subset17["Promotions"].mean()*(1.127**prd) 

    mon_subdf =  ABI_df.loc[(ABI_df["MI"] == month)& (ABI_df["err"] != "datacheck") ]            
    monTemp = mon_subdf.groupby("Agency").Max_Temp.mean()
    temperature = monTemp[eachA]
    
    #simple linear interpolation of Jan18 and Jan13 unit prices. ignores a small seasonality effect
    wt = period/60
    unitprice = Jan18_Unit_Price_AS[j,k]*wt + Jan13_Unit_Price_AS[j,k]*(1-wt)
    
    predict = np.dot(params,[1,period,unitprice,promote,temperature] ) 
    return predict
    
print("this will take a few minutes")
ABI_df["VolPred"] = ABI_df.apply(VolPredict,axis=1) #0 to 59

ABI_df["VolPred2"] = ABI_df["VolPred"]*(ABI_df["Seas"])
ABI_df["ratio"]= ABI_df["VolPred2"]/(ABI_df["Volume"])
ABI_df["resid1"]= (ABI_df["Volume"]) - ABI_df["VolPred2"]*1.0007


#Study Holiday Effect

#now regress the Vol  Vol Predict and calendary dummy vars
#missing=drop sidnt work for me
subset_df=ABI_df.loc[~np.isnan(ABI_df["ratio"]) & (ABI_df["err"] != "datacheck") ]

result = sm.ols(formula= "ratio ~ Easter_Day + Independence_Day + RevolutionDayMem + LaborDay + GoodFriday + NewYear + Xmas + Regional_Games + Beer_Capital", data=subset_df).fit()
result = sm.ols(formula= "Volume ~ VolPred2 + Easter_Day + Independence_Day + RevolutionDayMem + LaborDay + GoodFriday + NewYear + Xmas + Regional_Games + Beer_Capital", data=subset_df).fit()
result = sm.ols(formula= "Volume ~ VolPred2 + NewYear + Xmas", data=subset_df).fit()

result = sm.ols(formula= "ratio ~ Xmas + NewYear", data=subset_df).fit()

result = sm.ols(formula= "resid1 ~ NewYear + Xmas", data=subset_df).fit()
result = sm.ols(formula= "resid1 ~ NewYear + Xmas+Sales2HL+ Promotions+ Price_HL+ Max_Temp+ Soda_Volume+indy_Volume", data=subset_df).fit()
result = sm.ols(formula= "resid1 ~ NewYear + Xmas+Sales2HL+  Max_Temp+ Soda_Volume+indy_Volume", data=subset_df).fit()
result = sm.ols(formula= "resid1 ~ NewYear + Xmas+Sales2HL+  Max_Temp+ indy_Volume", data=subset_df).fit()
result = sm.ols(formula= "resid1 ~ NewYear + Xmas+Sales2HL+  Max_Temp", data=subset_df).fit() #Second Order Effects Regression


print(result.summary())

SecondOrderEff = np.zeros((58,25,13)) #AS3WAY with second order effects

#the 13th col in third dimension is the prediciton for Jan2018
for k, eachS in enumerate(SKU_List):
    for j,eachA in enumerate(agencyList):
        if countAS[j,k]>0:
            AS_subset =  ABI_df.loc[(ABI_df["Agency"] == eachA) & (ABI_df["SKU"] == eachS)& (ABI_df["err"] != "datacheck") ]
            AS_subset17=AS_subset.loc[AS_subset["YY"] > 2016.8]             
            if countAS[j,k]>40:
                result = sm.ols(formula= "SeasAdjVol ~ MQ + Sales2HL + Promotions + Max_Temp", data=AS_subset).fit()
                SecondOrderEff[j,k,:5] = result.params
                SecondOrderEff[j,k,5] = AS_subset.shape[0]
                SecondOrderEff[j,k,6] = result.rsquared
                SecondOrderEff[j,k,7:12] = result.pvalues
                promote=AS_subset17["Promotions"].mean()*1.151 
                temperature= JanTemp[eachA]
                unitprice= Jan18_Unit_Price_AS[j,k]
                #promotions incrd 27% '17 vs '16 and even faster in '16. 7months vs 12 . wont necessarily grow at same rate
                SAV =  np.dot(result.params,[1,60,unitprice,promote,temperature] ) 
                #if j==3 and k==3:                     print (result.summary())
                #from second order effects regression above
                SecondOrderEff[j,k,12] = SAV * SeasL[0] * 1.0007 +27.8704 +14.7021*temperature - .2474* unitprice
                #90.8497 nyday less 62.9793 slope

            else: #for AS pairs with short history, ratio off 2017 
                #on average these markets saw SAV decline 45%. I prorate this for partial yearperhaps some SKUs were abandoned
                SecondOrderEff[j,k,:12] = np.nan
                SecondOrderEff[j,k,12] = AS_subset17["SeasAdjVol"].mean()  *.71
            
        else:
            SecondOrderEff[j,k,:] = np.nan


Jan18_Volume_Pred2 = SecondOrderEff[:,:,12]

with open ("volume forecast V2.csv","w") as outFile:
    for k, eachS in enumerate(SKU_List):
        for j,eachA in enumerate(agencyList):
            volume= Jan18_Volume_Pred2[j,k]
            if np.isnan(volume) or volume<0:
                outFile.write("%s,%s,0\n" % (eachA,eachS))
            else:
                outFile.write("%s,%s,%0.2f\n" % (eachA,eachS,Jan18_Volume_Pred2[j,k] ))

subset2_df=ABI_df.loc[~np.isnan(ABI_df["ratio"]) & (ABI_df["MQ"] > 48 ) & (ABI_df["err"] != "datacheck") ]

result = sm.ols(formula= "resid1 ~ NewYear + Xmas+Sales2HL+  Max_Temp", data=subset2_df).fit()
print(result.summary())


RevSecondOrderEff = np.zeros((58,25,13)) #AS3WAY with second order effects

#the 13th col in third dimension is the prediciton for Jan2018
for k, eachS in enumerate(SKU_List):
    for j,eachA in enumerate(agencyList):
        if countAS[j,k]>0:
            AS_subset =  ABI_df.loc[(ABI_df["Agency"] == eachA) & (ABI_df["SKU"] == eachS)& (ABI_df["err"] != "datacheck") ]
            AS_subset17=AS_subset.loc[AS_subset["YY"] > 2016.8]             
            if countAS[j,k]>40:
                result = sm.ols(formula= "SeasAdjVol ~ MQ + Sales2HL + Promotions + Max_Temp", data=AS_subset).fit()
                RevSecondOrderEff[j,k,:5] = result.params
                RevSecondOrderEff[j,k,5] = AS_subset.shape[0]
                RevSecondOrderEff[j,k,6] = result.rsquared
                RevSecondOrderEff[j,k,7:12] = result.pvalues
                promote=AS_subset17["Promotions"].mean()*1.151 
                temperature= JanTemp[eachA]
                unitprice= Jan18_Unit_Price_AS[j,k]
                #promotions incrd 27% '17 vs '16 and even faster in '16. 7months vs 12 . wont necessarily grow at same rate
                SAV =  np.dot(result.params,[1,60,unitprice,promote,temperature] ) 
                #if j==3 and k==3:                     print (result.summary())
                #from second order effects regression above
                RevSecondOrderEff[j,k,12] = SAV * SeasL[0] * 1.0007 -120.2 +5.1768*temperature -0.0165* unitprice
                #90.8497 nyday less 62.9793 slope

            else: #for AS pairs with short history, ratio off 2017 
                #on average these markets saw SAV decline 45%. I prorate this for partial yearperhaps some SKUs were abandoned
                RevSecondOrderEff[j,k,:12] = np.nan
                RevSecondOrderEff[j,k,12] = AS_subset17["SeasAdjVol"].mean()  *.71
            
        else:
            RevSecondOrderEff[j,k,:] = np.nan
np.nanmean(RevSecondOrderEff,axis=(0,1))


Jan18_Volume_Pred3 = RevSecondOrderEff[:,:,12]
with open ("volume forecast V3.csv","w") as outFile:
    for k, eachS in enumerate(SKU_List):
        for j,eachA in enumerate(agencyList):
            volume= Jan18_Volume_Pred3[j,k]
            if np.isnan(volume) or volume<0:
                outFile.write("%s,%s,0\n" % (eachA,eachS))
            else:
                outFile.write("%s,%s,%0.2f\n" % (eachA,eachS,Jan18_Volume_Pred3[j,k] ))


#prints history for oone of the largest Agency SKU pairs to faciliate comparion in excel file
testcase =  ABI_df.loc[(ABI_df["Agency"] == "Agency_02") & (ABI_df["SKU"] == "SKU_03")& (ABI_df["err"] != "datacheck") ]
testcase.groupby("YY").mean()

ABI_df["SARev"] =ABI_df["Sales2HL"]*ABI_df["SeasAdjVol"]
ABI_df["SAPCRev"] =ABI_df["SARev"]/ABI_df["avg_pop17"]

bySKU = ABI_df.groupby("SKU").mean()["SAPCRev"]


#Analyis of Key Products
MajorSKU = ["SKU_05","SKU_01","SKU_04","SKU_02","SKU_03","SKU_06","SKU_22","SKU_11","SKU_18","SKU_08","SKU_32","SKU_34"]

HHI06,HHI14,Temp06,Temp14 = 228353, 204289, 29.008, 25.085
#6 is Warmer and mre prosperous
#Tempearture 06 29.0079391 +-	2.649823581	 	25.08528079 +-1.968971985


for eachS in MajorSKU:
        AS_subset =  ABI_df.loc[(ABI_df["SKU"] == eachS)& (ABI_df["err"] != "datacheck") ]
        result = sm.ols(formula= "SAPCRev ~ MQ + avg_HHI17 + Max_Temp", data=AS_subset).fit()
        predict06 =  np.dot(result.params,[1,96, HHI06, Temp06] ) 
        predict14 =  np.dot(result.params,[1,96, HHI14, Temp14] ) 
        print(eachS, predict06, predict14, result.params)

#we need the covariance matrix


Special= ["Agency_33","Agency_34","Agency_40","Agency_36","Agency_35","Agency_44","Agency_37"]

for eachA in Special:
    testcase2 =  ABI_df.loc[(ABI_df["Agency"] == eachA)& (ABI_df["err"] != "datacheck") ]
    print(eachA,testcase2.groupby("SKU").SAPCRev.mean())






