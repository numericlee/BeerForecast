import pandas as pd, numpy as np

#this module integrates data from various files

PATH = r"Yourpath\DS\abinbev\givens"
DF2 = pd.read_csv(PATH+  "\historical_volume.csv", header=0,sep=",")
DF = DF2.set_index(["Agency","SKU","YearMonth"]).sort_index()

HolidayName_edit=["Easter_Day","GoodFriday","NewYear","Xmas","LaborDay",
                  "Independence_Day","RevolutionDayMem", "Regional_Games",
                  "FIFA_U17_WCup","Football_Gold_Cup","Beer_Capital","MusicFest"]

DemogX = ["avg_pop17","avg_HHI17"]
PSPX = ["Price_HL","Sales2HL","Promotions"]
newcols=["Max_Temp","Soda_Volume","indy_Volume","Count"]+DemogX+ PSPX + HolidayName_edit

DF[newcols]=pd.DataFrame([[np.nan]*len(newcols)])
DF["err"] = ""

AgencyList =  DF2["Agency"].unique()
YearMonthList = DF2["YearMonth"].unique()

indsodsales = pd.read_csv(PATH+"\industry_soda_sales.csv", header=0,sep=",").set_index("YearMonth")   #
indvol = pd.read_csv(PATH+"\industry_volume.csv", header=0,sep=",").set_index("YearMonth")   #

for YYYYMM in YearMonthList:     
    DF.loc [(slice(None),slice(None),YYYYMM),  "Soda_Volume"] = indsodsales.loc[YYYYMM, "Soda_Volume"]   
    DF.loc [(slice(None),slice(None),YYYYMM), "indy_Volume"] =      indvol.loc[YYYYMM, "Industry_Volume"]
    

demog = pd.read_csv(PATH+"\demographics.csv", header=0,sep=",").set_index("Agency")   #
evtcdr = pd.read_csv(PATH+"\event_calendar.csv", header=0,sep=",").set_index("YearMonth")   #
weather = pd.read_csv(PATH+"\weather.csv", header=0,sep=",").set_index(["Agency","YearMonth"])   #

psp0 = pd.read_csv(PATH+"\price_sales_promotion.csv", header=0,sep=",").set_index(["Agency","SKU","YearMonth"])
psp = psp0.sort_index()
    
for Agency in AgencyList:
    
    DF.loc[(Agency,slice(None),slice(None)),DemogX] = demog.loc[Agency].values
    
    for YYYYMM in YearMonthList:
        
        DF.loc[(Agency,slice(None),YYYYMM),"Max_Temp"] = weather.loc[Agency,YYYYMM]["Avg_Max_Temp"] 

        SKUlist= DF.loc[Agency,slice(None), YYYYMM].index.get_level_values('SKU').unique()
        for SKU in SKUlist:
            idx= (Agency,SKU,YYYYMM)
            DF.loc[idx,PSPX] = psp.loc[Agency,SKU,YYYYMM].values 
            DF.loc[idx, HolidayName_edit] =  evtcdr.loc[YYYYMM].values 
    print("_", Agency)
 

DF.drop(DF[(DF["Volume"]==0)  & (DF["Sales2HL"]==0) ].index, inplace=True)
DF["Rev"] = DF["Sales2HL"] * DF["Volume"]
DF["Count"] = 1

DF.loc[ (DF["Volume"]==0) |(DF["Sales2HL"]==0) | (DF["Sales2HL"] > DF["Price_HL"]) , "err"   ] = "DataCheck"

DF.to_csv("ABIq.csv",sep="|") 




