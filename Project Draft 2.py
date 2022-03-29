#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1. Project Overview


# In[2]:


#This Project aims to investigate the calculation of player expected goals (xG).
#The Project will look at data on premier league players over the last three seasons imported from "fb ref"
#fb ref's describe xG as being "the probability that a shot will result in a goal based on the characteristics of that shot and the events leading up to it. Some of these characteristics/variables include:
#Location of shooter: How far was it from the goal and at what angle on the pitch?
#Body part: Was it a header or off the shooter's foot?
#Type of pass: Was it from a through ball, cross, set piece, etc?
#Type of attack: Was it from an established possession? Was it off a rebound? Did the defense have time to get in position? Did it follow a dribble?"
#This project aims to explore areas in which the accuracy of this calculation can improve


# In[3]:


# 2. Importing, Cleaning & Merging Data


# In[4]:


#imported relevant packages and set custom settings for the project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[5]:


#Imported 21/22 player data

prem_data_raw_this_season = pd.read_csv(r"C:\Users\connollyn\Documents\Prem_Player_Data.csv",index_col="Rk")


# In[6]:


#Quick look at the data that was imported and check it was imported correctly

prem_data_raw_this_season.head()


# In[7]:


#Overview of the number of columns & rows in the data set

prem_data_raw_this_season.shape


# In[8]:


#Here we can see the extra blank rows where imported at the end of the data set

prem_data_raw_this_season.tail()


# In[9]:


#Lets take a look at the full list of variables we have to work with in the dataset

for col in prem_data_raw_this_season.columns:
    print(col)


# In[10]:


#An explanation of the acronyms used in some of the key columns of data imported from fb ref

#Gls = Goals Scored
#Sh = Shots (Does not include penalty kicks)
#SoT = Shots on target (Does not include penaly kicks)
#Dist = Average distance in yards from goal of all shots taken (Does not include penalty kicks)
#FK = Shots from free kicks
#PK = Penalty kicks scored
#PKatt = Penalty kicks taken
#npxG = Non-penalty expected goals
#G-xG = Goals - expected goals
#np:G-xG = Non-penalty goals - non penalty expexcted goals


# In[11]:


#Next the columns which wont be analysed in the project are dropped

prem_data_this_season = prem_data_raw_this_season.drop(columns=["Nation","Age","Born","Matches","SoT%","Sh/90","SoT/90","G/Sh","G/SoT","npxG/Sh"])


# In[12]:


#Rows with missing data in these key metrics are also dropped

prem_data_this_season = prem_data_this_season.dropna(subset=["Player","Pos","Squad","90s","Sh"])


# In[13]:


#To ensure high data integrity we drop duplicates, this avoids players who transferred mid season being counted twice

prem_data_this_season.drop_duplicates(subset=["Player"], keep=False, inplace = True)


# In[14]:


#A look at how the data cleaning has affected the dataframe

prem_data_this_season.shape


# In[15]:


#Next players who have not had a shot on target are removed as they're data for this season will not be any use to this project

prem_data_this_season["Valid_Data"] = prem_data_this_season["Sh"]>0
prem_data_this_season = prem_data_this_season[prem_data_this_season["Valid_Data"] != False]
prem_data_this_season = prem_data_this_season.drop(columns=["Valid_Data"])


# In[16]:


#Next we import the data for the 20/21 & 19/20 seasons

prem_data_20_21_raw = pd.read_csv(r"C:\Users\connollyn\Documents\Prem_Player_Data_20_21.csv", index_col="Rk")
prem_data_19_20_raw = pd.read_csv(r"C:\Users\connollyn\Downloads\Prem_Player_Data_19_20.csv", index_col="Rk")


# In[17]:


#Based on the steps that were required to clean the data for the 21/22 season a function is defined to quickly & consistently clean the data for the 20/21 & 19/20 seasons

def Clean_Data(df):
    df2 = df.drop(columns=["Nation","Age","Born","Matches","SoT%","Sh/90","SoT/90","G/Sh","G/SoT","npxG/Sh"])
    df3 = df2.dropna(subset=["Player","Pos","Squad","90s","Sh"])
    df3.drop_duplicates(subset=["Player"], keep = False, inplace = True)
    df3["Valid_Data"] = df3.loc[:,"Sh"]>0
    df4 = df3[df3["Valid_Data"] != False]
    df5 = df4.drop(columns=["Valid_Data"])
    return df5


# In[18]:


#The function is applied to clean the data

prem_data_20_21 = Clean_Data(prem_data_20_21_raw)
prem_data_19_20 = Clean_Data(prem_data_19_20_raw)


# In[19]:


#The function has left 419 valid rows for 20/21 data

prem_data_20_21.shape


# In[20]:


#The function has left 419 valid rows for 19/20 data

prem_data_19_20.shape


# In[21]:


#The 21/22 & 20/21 data is merged

prem_data_two_seasons = prem_data_this_season.merge(right=prem_data_20_21, how="outer", on="Player", suffixes = ("_21_22","_20_21"))


# In[22]:


#The data for the 19/20 season is also merged

prem_data_19_20 = prem_data_19_20.add_suffix("_19_20")
prem_data_19_20.rename(columns={"Player_19_20":"Player"}, inplace = True)
prem_data_three_seasons_raw = prem_data_two_seasons.merge(right=prem_data_19_20,how="outer",on="Player")


# In[23]:


#There is now a large number of columns in the merged dataframe

for col in prem_data_three_seasons_raw.columns:
    print(col)


# In[24]:


#To reduce the number of columns we will total the data for key variables across three seasons

prem_data_three_seasons_raw = prem_data_three_seasons_raw.fillna(value=0)
prem_data_three_seasons_raw["Total_90's"] = prem_data_three_seasons_raw["90s_21_22"] +  prem_data_three_seasons_raw["90s_20_21"] +  prem_data_three_seasons_raw["90s_19_20"]
prem_data_three_seasons_raw = prem_data_three_seasons_raw.drop(columns=["90s_21_22","90s_20_21","90s_19_20"])


# In[25]:


#Lets ensure its had the desired affect

prem_data_three_seasons_raw.head()


# In[26]:


#A custom function is defined to total the remaining variables

def Total_Data(df, Col1, Col2, Col3,Total):
    df[Total] = df[Col1] + df[Col2] + df[Col3]
    df2 = df.drop(columns=[Col1,Col2,Col3])
    return df2
    
    


# In[27]:


#Before we apply the function we must adjust the distance for each year so the total average distance will be apropriately weighted

prem_data_three_seasons_raw["Total_Dist_21_22"] = prem_data_three_seasons_raw["Dist_21_22"] * prem_data_three_seasons_raw["Sh_21_22"]
prem_data_three_seasons_raw["Total_Dist_20_21"] = prem_data_three_seasons_raw["Dist_20_21"] * prem_data_three_seasons_raw["Sh_20_21"]
prem_data_three_seasons_raw["Total_Dist_19_20"] = prem_data_three_seasons_raw["Dist_19_20"] * prem_data_three_seasons_raw["Sh_19_20"]
prem_data_three_seasons_raw = prem_data_three_seasons_raw.drop(columns=["Dist_21_22", "Dist_20_21", "Dist_19_20"])


# In[28]:


#The function is applied

prem_data_three_seasons_raw = Total_Data(prem_data_three_seasons_raw, "SoT_21_22", "SoT_20_21", "SoT_19_20", "Total_SoTs")
prem_data_three_seasons_raw = Total_Data(prem_data_three_seasons_raw, "Gls_21_22", "Gls_20_21", "Gls_19_20", "Total_Gls")
prem_data_three_seasons_raw = Total_Data(prem_data_three_seasons_raw, "Sh_21_22", "Sh_20_21", "Sh_19_20", "Total_Sh")
prem_data_three_seasons_raw = Total_Data(prem_data_three_seasons_raw, "FK_21_22", "FK_20_21", "FK_19_20", "Total_FKs")
prem_data_three_seasons_raw = Total_Data(prem_data_three_seasons_raw, "PK_21_22", "PK_20_21", "PK_19_20", "Total_PKs")
prem_data_three_seasons_raw = Total_Data(prem_data_three_seasons_raw, "PKatt_21_22", "PKatt_20_21", "PKatt_19_20", "Total_PKatt")
prem_data_three_seasons_raw = Total_Data(prem_data_three_seasons_raw, "xG_21_22", "xG_20_21", "xG_19_20", "Total_xG")
prem_data_three_seasons_raw = Total_Data(prem_data_three_seasons_raw, "npxG_21_22", "npxG_20_21", "npxG_19_20", "Total_npxG")
prem_data_three_seasons_raw = Total_Data(prem_data_three_seasons_raw, "G-xG_21_22", "G-xG_20_21", "G-xG_19_20", "Total_G-xG")
prem_data_three_seasons_raw = Total_Data(prem_data_three_seasons_raw, "np:G-xG_21_22", "np:G-xG_20_21", "np:G-xG_19_20", "Total_np:G-xG")
prem_data_three_seasons_raw = Total_Data(prem_data_three_seasons_raw, "Total_Dist_21_22", "Total_Dist_20_21", "Total_Dist_19_20", "Total_Dist")


# In[29]:


#The average distance is calcualted

prem_data_three_seasons_raw["Avg_Dist"] = prem_data_three_seasons_raw["Total_Dist"] / prem_data_three_seasons_raw["Total_Sh"]
prem_data_three_seasons_raw = prem_data_three_seasons_raw.drop(columns = ["Total_Dist"])


# In[30]:


#The columns have been largely reduced

for col in prem_data_three_seasons_raw.columns:
    print(col)


# In[31]:


#Players who have a small sample size of minutes or shots are dropped as they are not of much use to the project

prem_data_three_seasons_raw["Valid_Data"] = prem_data_three_seasons_raw["Total_90's"] >=5 
prem_data_three_seasons_raw["Valid_Data_2"] = prem_data_three_seasons_raw["Total_Sh"] >=5
prem_data_three_seasons_raw = prem_data_three_seasons_raw[prem_data_three_seasons_raw["Valid_Data"] != False]
prem_data_three_seasons_raw = prem_data_three_seasons_raw[prem_data_three_seasons_raw["Valid_Data_2"] != False]
prem_data_three_seasons = prem_data_three_seasons_raw.drop(columns = ["Valid_Data", "Valid_Data_2"])


# In[32]:


# 3. Analysis & Visualisation


# In[33]:


#The dataframe is sorted by the number of minutes the players have played & the index is changed to be informative

prem_data_three_seasons.sort_values(by="Total_90's", ascending=False, inplace = True)
prem_data_three_seasons.reset_index(inplace = True)
prem_data_three_seasons.index.rename("Rank_For_Total_90's", inplace = True)
prem_data_three_seasons = prem_data_three_seasons.drop(columns = ["index"])
prem_data_three_seasons.index = prem_data_three_seasons.index + 1


# In[34]:


#The players who have played the most over the last three years can be seen

prem_data_three_seasons.head(10)


# In[35]:


#The dataframe is now sorted by xG as & we can see the players who have the highest xG over the last 3 years

prem_data_three_seasons.sort_values(by="Total_xG", ascending=False, inplace = True)
prem_data_three_seasons.head(10)


# In[36]:


#A column is added to state a players non-penalty goals scored as this is a useful variable that is missing

prem_data_three_seasons["Total_Np_Gls"] = prem_data_three_seasons["Total_Gls"] - prem_data_three_seasons["Total_PKs"]


# In[37]:


#The performance of npxG against non-penalty goals is reviewed by position for 21/22

prem_data_three_seasons.groupby(["Pos_21_22"]).agg({"Total_np:G-xG":["mean","min","max","sum","count"]})


# In[38]:


#The performance of npxG against non-penalty goals in 20/21 is reviewed by position for players who didnt play in 21/22

No_21_22_pos = prem_data_three_seasons[(prem_data_three_seasons["Pos_21_22"] == 0)]


# In[39]:


No_21_22_pos.groupby(["Pos_20_21"]).agg({"Total_np:G-xG":["mean","min","max","sum","count"]})


# In[40]:


#The performance of npxG against non-penalty goals in 20/21 is reviewed by position for players who didnt play in 21/22

No_21_22_or_20_21_pos = No_21_22_pos[(No_21_22_pos["Pos_20_21"] == 0)]


# In[41]:


No_21_22_or_20_21_pos.groupby(["Pos_19_20"]).agg({"Total_np:G-xG":["mean","min","max","sum","count"]})


# In[42]:


#The average difference between non-penalty expected goals & non-penalty goals is grouped in a list with the sum of players of for whom that is their last position played

Position_npxg_performance = {"DF":[(-1.2 + -7.5 + -3.9)/(115 + 39 + 19),(115 + 39 + 19)], "DF_FW":[(-2.5 + 1.2)/(5 + 2),(5 + 2)], "DF_MF":[(-3.6 + -2.1 + -1)/(5 + 3 +1 ),(5 + 3 +1 )], "FW":[(-7.6 + -8.9 + -6.3)/(52 + 16 + 8),(52 + 16 + 8)], "FW_DF":[-0.4, 1], "FW_MF":[(14.2 + -6.4 + -3.7)/(47 + 14 + 5),(47 + 14 + 5)], "MF":[(-1.1 + -8.4 + -8.7)/(91 + 24 + 18),(91 + 24 + 18)], "MF_DF": [(-6.9 + -0.3 + 1.4)/(5 + 2 + 2),(5 + 2 + 2)], "MF_FW": [(-2.7 + 6.9 + -1.9)/(27 + 8 + 6),(27 + 8 + 6)]}


# In[43]:


#The below boxplot shows that the difference must be expressed as a percentage of total npxG as total npxG is largely different by position
#On a seperate note the data has a large spread with lots of outliers which will make the median a desirable metric for certain calculations going forward

sns.set_style("whitegrid")
sns.catplot(data=prem_data_three_seasons, x = "Pos_21_22", y= "Total_npxG", kind = "box")
plt.suptitle("xG by postion")
plt.xticks(rotation=90)
plt.show()


# In[44]:


#The npxG for 21/22 is shown by position

prem_data_three_seasons.groupby(["Pos_21_22"]).agg({"Total_npxG":["mean","min","max","sum","count"]})


# In[45]:


#The npxG for 20/21 is shown by position

No_21_22_pos.groupby(["Pos_20_21"]).agg({"Total_npxG":["mean","min","max","sum","count"]})


# In[46]:


#The npxG for 19/20 is shown by position

No_21_22_or_20_21_pos.groupby(["Pos_19_20"]).agg({"Total_npxG":["mean","min","max","sum","count"]})


# In[47]:


#The total npxG by players most recent position is added to a list

Total_npxg_by_pos = {"DF":(267.2 + 55.5 + 13.9), "DF_FW":(9.5 + 0.8), "DF_MF":(19.6 + 2.1 + 1), "FW":(828.6 + 121.9 + 24.3), "FW_DF": 2.4, "FW_MF":(401.8 + 70.4 + 10.7), "MF":(332.1 + 48.4 + 23.7), "MF_DF":(13.9 + 6.3 + 1.6), "MF_FW":(190.7 + 16.1 + 12.9)}


# In[48]:


#The npxG by position & difference between non-penalty xG & non-penalty goals has been merged into one dataframe

Position_npxg_performance_df = pd.DataFrame.from_dict(Position_npxg_performance, orient = "index", columns = ["mean_np:G-xG","count"])
Total_npxg_by_pos_df = pd.DataFrame.from_dict(Total_npxg_by_pos, orient = "index", columns = ["total_npxg"])
pos_npxg_perf = Position_npxg_performance_df.merge(right = Total_npxg_by_pos_df, how = "outer", left_index = True, right_index = True)
pos_npxg_perf["percent_npxG_diff"] = (pos_npxg_perf["mean_np:G-xG"]/(pos_npxg_perf["total_npxg"]/pos_npxg_perf["count"]))*100


# In[49]:


#The percentage difference by position is expressed below

pos_npxg_perf


# In[50]:


#The average difference in non-penalty goals versus non penalty goals irrespective of position is 2.48%, We can see from the previous table that defenders & midfielders have both performed worse than this

np.mean(prem_data_three_seasons["Total_np:G-xG"])/np.mean(prem_data_three_seasons["Total_npxG"]) * 100


# In[51]:


#The performance of npxG against non penalty goals is visualised below

sns.relplot(data=prem_data_three_seasons, x="Total_Np_Gls", y="Total_npxG", row="Pos_21_22", kind = "scatter", )
plt.suptitle("Non-penalty xg prediction success by position", y =1.03)
plt.show()


# In[52]:


#As the percentage difference between goals & xG is considerably lower this indicates that the model may perform well in assessing the value of penalties

np.mean(prem_data_three_seasons["Total_G-xG"])/np.mean(prem_data_three_seasons["Total_xG"]) * 100


# In[53]:


#The assumed correlation between shots & goals excluding penalties is shown below

np.corrcoef(prem_data_three_seasons["Total_Sh"],prem_data_three_seasons["Total_npxG"])


# In[54]:


#The actual correlation between shots & goals excluding penalties is shown below

np.corrcoef(prem_data_three_seasons["Total_Sh"],prem_data_three_seasons["Total_Np_Gls"])


# In[55]:


#The assumed correlation between shots on target & goals excluding penalties is shown below

np.corrcoef(prem_data_three_seasons["Total_SoTs"],prem_data_three_seasons["Total_npxG"])


# In[56]:


#The actual correlation between shots on target & goals excluding penalties is shown below

np.corrcoef(prem_data_three_seasons["Total_SoTs"],prem_data_three_seasons["Total_Np_Gls"])


# In[57]:


#This indicates the model may overestimate the value of shots off target


# In[58]:


#The averages from players in the dataframe is prepared to be visualised

Metric_Averages = {"SoT_mean": prem_data_three_seasons["Total_SoTs"].mean()}
Metric_Averages["Sh_mean"] = prem_data_three_seasons["Total_Sh"].mean()
Metric_Averages["Gls_mean"] = prem_data_three_seasons["Total_Gls"].mean()
Metric_Averages["Np_Gls_mean"] = prem_data_three_seasons["Total_Np_Gls"].mean()
Metric_Averages["xG_mean"] = prem_data_three_seasons["Total_xG"].mean()
Metric_Averages["npxG_mean"] = prem_data_three_seasons["Total_npxG"].mean()


# In[59]:


Metric_Averages_df = pd.DataFrame.from_dict(Metric_Averages, orient = "index", columns = ["Mean"])


# In[60]:


Metric_Averages_df.reset_index(inplace = True)


# In[61]:


#The visualisation again shows that the average xG is slightly overestimated, It also demonstrates that are only a small percentage of shots translate to goals

sns.catplot(data = Metric_Averages_df, x = "index", y = "Mean", kind = "bar")
sns.set_context("poster")
plt.suptitle("Player averages", y =1.05)
plt.xlabel("Key metrics")
plt.xticks(rotation = 90)
plt.show()


# In[62]:


#The exact values of the above chart are shown

Metric_Averages_df


# In[63]:


#shots on target per non penalty goals
15.91/4.69


# In[64]:


#shots per non penalty goals
47.35/4.69


# In[65]:


#shots on target per non penalty xg 
15.91/4.8


# In[66]:


#shots per non penalty xg
47.35/4.8


# In[67]:


#Overestimation of sot value
((3.39/3.31)-1)*100


# In[68]:


#Overestimation of sh value
((10.1/9.86)-1)*100


# In[69]:


#Next the xG performance is reviewed for penalty takers
#Regualar penalty takers appear to have an increased overperformance of goals versus xG
#Earlier we concluded that xG is calculated more accuarately when penalties are included so perhaps this is more at play here

sns.relplot(data=prem_data_three_seasons, x = "Total_PKatt", y = "Total_G-xG", kind = "scatter")
plt.suptitle("xG accuaracy for penalty takers",y=1.05)
plt.show()


# In[70]:


#Regular penalty takers also appear to be overperforming in npxG
#These players are likely to be overperforming in terms of xG as they are very good finishers and hence the take the teams penalties
#The model could potentially be improved if it took into account the players quality and finishing ability

sns.relplot(data=prem_data_three_seasons, x = "Total_PKatt", y = "Total_np:G-xG", kind = "scatter")
plt.suptitle("npxG accuaracy for penalty takers",y=1.05)
plt.show()


# In[71]:


#To further investigate players are catagorised as either penalty takers or not penalty takers

prem_data_three_seasons["PK_takers"] = prem_data_three_seasons["Total_PKatt"]>=5
penalty_takers = prem_data_three_seasons[prem_data_three_seasons["PK_takers"] != False]


# In[72]:


#The Sample of 22 players provides 

penalty_takers.agg({"Total_G-xG" :["sum","mean","count","median"], "Total_np:G-xG" : ["sum","mean","count","median"], "Total_xG" : ["sum","mean","count","median"], "Total_npxG" : ["sum","mean","count","median"]})


# In[73]:


#Penalty takers have overperformed in terms of non-penalty goals versus npxG by 10.86% which is a considerable difference to the 2.48% underperformance of the average player in the dataframe 

(2.15/19.8) * 100


# In[74]:


#On the basis that the model may underestimate players of quality, players from the "big six" clubs are grouped together

Ars = prem_data_three_seasons[prem_data_three_seasons["Squad_21_22"] == "Arsenal"]
Che = prem_data_three_seasons[prem_data_three_seasons["Squad_21_22"] == "Chelsea"]
Liv = prem_data_three_seasons[prem_data_three_seasons["Squad_21_22"] == "Liverpool"]
MU = prem_data_three_seasons[prem_data_three_seasons["Squad_21_22"] == "Manchester Utd"]
ManC = prem_data_three_seasons[prem_data_three_seasons["Squad_21_22"] == "Manchester City"]
Tot = prem_data_three_seasons[prem_data_three_seasons["Squad_21_22"] == "Tottenham"]
Big_Six_Players = pd.concat([Ars, Che, Liv, MU, ManC, Tot])


# In[75]:


#The perfermonce of npxG as a predicter in the big six is shown below

sns.relplot(data=Big_Six_Players, x="Total_Np_Gls", y="Total_npxG", hue = "Squad_21_22", kind="scatter")
plt.suptitle("Accuracy of npxG in big six players", y=1.05)
plt.show()


# In[76]:


#We can see the correlation between shots & goals excluding penalties is stronger in players playing with the big six

np.corrcoef(Big_Six_Players["Total_Sh"],Big_Six_Players["Total_Np_Gls"])


# In[77]:


#We can see the correlation between shots on target & goals excluding penalties is stronger in players playing with the big six

np.corrcoef(Big_Six_Players["Total_SoTs"],Big_Six_Players["Total_Np_Gls"])


# In[78]:


#We can also see that players in the big six have overperformed in goals/non-penalty goals versus xG/npxG even though the average player has underperformed

Big_Six_Players.agg({"Total_G-xG" :["sum","mean","count","median"], "Total_np:G-xG" : ["sum","mean","count","median"]})


# In[79]:


#Next to analyse player quality the players who have topped xG over the last season are subgrouped as they are likely high quality players

top_xG_performers = prem_data_three_seasons.head(50)


# In[80]:


#Here we can see that these quality players have also been overperforming against xG

top_xG_performers.agg({"Total_G-xG" :["sum","mean","count","median"], "Total_np:G-xG" : ["sum","mean","count","median"]})


# In[81]:


#The Dataframe is now sorted to view the best & worst performers against non-penalty xG in the dataframe

prem_data_three_seasons.sort_values(by="Total_np:G-xG", ascending=False, inplace = True)


# In[82]:


prem_data_three_seasons


# In[83]:


#Whether the player takes free kicks or not is added to the dataframe as a boolean

prem_data_three_seasons["Free_kick_taker"] = prem_data_three_seasons["Total_FKs"]>=5


# In[84]:


#Next we categorise the distance players like to shoot from as long, medium & short

np.percentile(prem_data_three_seasons["Avg_Dist"], [66,33])


# In[85]:


conditions = [prem_data_three_seasons["Avg_Dist"] >= 18.89, prem_data_three_seasons["Avg_Dist"] <= 14.85]
Values = ["Long", "Short"]


# In[86]:


prem_data_three_seasons["Shooting_habits"] = np.select(conditions, Values, default = "Medium")


# In[87]:


#Good and bad finishers are grouped based on their performance against non-penalty xG

Clinical_finishers = prem_data_three_seasons.head(30)
Bad_finishers = prem_data_three_seasons.tail(30)


# In[88]:


#The characteristics of good finishers are visualised below

sns.catplot(data = Clinical_finishers, x="PK_takers", col="Free_kick_taker", hue="Shooting_habits", hue_order = ["Short","Medium","Long"],kind="count")
plt.suptitle("Characteristics of a clinical finisher", y=1.05)
plt.show()


# In[89]:


#The characteristics of bad finishers are visualised below

sns.catplot(data = Bad_finishers, x="PK_takers", col="Free_kick_taker", hue="Shooting_habits",kind="count")
plt.suptitle("Characteristics of a bad finisher",y=1.05)
plt.show()


# In[90]:


#As we can see in the charts above, xG overperformers are much more likely to take free kicks & penalties than the underperformers likely due to there shooting ability
#Overperformers also tend to take shots from longer distances than underperformers
#This is either a result of high confidence in their shooting or an underestimation of the value of shots from distance in the xG model


# In[91]:


# 4. Conclusion


# In[92]:


#The xG model operates under the assumption that all men are created equal (which isn't quite the case on the football pitch)
#5 major conclusions about fb ref's xG model are outlined below


# In[93]:


#1. The position a player plays on the field should be given some weight

#As we can see from the analysis of players by their most recent position, forwards performed a lot better against their xG than other positions


# In[94]:


#2. The value of shots & shots on target are overestimated in the model

#The value of these metrics are overstated by about 2.4% each
#As displayed in the countplot "Player averages" npxG & xg is slighlty bigger than Gls & np_gls


# In[95]:


#3. The value of a shot in the xG model should be increased if it's by a regular penalty taker

#Players who take penalties are underestimated in the xG model
#Players who take penalties xG overperformance is clealy displayed in the scatter plot "xG accuaracy for penalty takers"
#Furthermore evidence that this is not due to underestimation of the value of penalties in the model is clearly displayed in he scatter plot "npxG accuaracy for penalty takers"
#This is backed by the fact penalty takers have overperformed in npxG by an average of 10.86%
#This is likely due to the fact they are very good finishers


# In[96]:


#4 The club a player plays for should be considered when valueing a shot in the xG model

#Players who play at one of the "big six" clubs perform better against xG
#The scatterplot "Accuracy of npxG in big six players" shows the points sloping more towards np_gls than npxG indicating an overperformance versus npxG
#This is supported in the quantitative analysis which has them overperforming against npxG even though the average of the rest of the dataset was underperforming by 2.48%


# In[97]:


#5 The value of a shot from free kick takers and longe range shooters should be increased in the model

#Players who overperform from xG are more likely to take set pieces & shoot from distance
#As shown in the countplots "Characteristics of a clinical finisher" & "Characteristics of a bad finisher" are much more likely to take free kicks & penalties
#Players on set pieces & who shoot from distance clearly have confidence in their shooting ability & for good cause


# In[98]:


# 5. Machine Learning

#This project could also be extended in the future to include machine learning
#Given the recommnedations listed in the above conclusion, the model could benefit largely from the application of machine learning
#My recommendation for the application of this machine learning would be to expand the data set by examing players across the top 5 european leagues
#Next segregate their finishing ability into subgroups based on position, whether they take penalties & or free kicks, the club they play for & the distance they like to shoot from
#From there regression analysis can be performed on each subgroup examining the relationship between their shots/shots on target and goals
#Withold 30% of the data for testing
#Once completed the value of a shot or shot on target in the xG model should be more accurately predicted based on the person who took it

