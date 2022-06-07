# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:25:38 2022

@author: Alex DelBene
"""
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np



# # # Read in data from CSV's into data frames and clean up # # #

co2_df = pd.read_csv('CO2_data.csv')  
co2_df = co2_df.drop(["Data source", "Sector", "Gas", "Unit"], axis = 1)



gdp_df = pd.read_csv('GDP_data.csv') 
gdp_df = gdp_df.drop(["Country Code", "Indicator Name", "Indicator Code"], axis = 1)
gdp_df = gdp_df.drop(gdp_df.iloc[:,1:31], axis = 1)
gdp_df = gdp_df.drop(gdp_df.iloc[:,30:32], axis = 1)



gdppc_df = pd.read_csv('GDP_per_capita_data.csv')  
gdppc_df = gdppc_df.drop(["Country Code", "Indicator Name", "Indicator Code"], axis = 1)
gdppc_df = gdppc_df.drop(gdppc_df.iloc[:,1:31], axis = 1)
gdppc_df = gdppc_df.drop(gdppc_df.iloc[:,30:32], axis = 1)



pop_df = pd.read_csv('Population_data.csv') 
pop_df = pop_df.drop(["Country Code", "Indicator Name", "Indicator Code"], axis = 1)
pop_df = pop_df.drop(pop_df.iloc[:,1:31], axis = 1)
pop_df = pop_df.drop(pop_df.iloc[:,30:32], axis = 1)





# # # Removing countries not shared between data sets/non-countries and swapping indices to country names # # # 

gdp_df = gdp_df[gdp_df["Country"].isin(co2_df["Country"])]
gdp_df = gdp_df.sort_values('Country')
gdp_df = gdp_df.reset_index(drop=True)



co2_df = co2_df[co2_df["Country"].isin(gdp_df["Country"])]
co2_df = co2_df.sort_values('Country')
co2_df = co2_df.reset_index(drop=True)



gdppc_df = gdppc_df[gdppc_df["Country"].isin(gdp_df["Country"])]
gdppc_df = gdppc_df.sort_values('Country')
gdppc_df = gdppc_df.reset_index(drop=True)



pop_df = pop_df[pop_df["Country"].isin(gdp_df["Country"])]
pop_df = pop_df.sort_values('Country')
pop_df = pop_df.reset_index(drop=True)



co2_df.index = co2_df.Country
gdp_df.index = gdp_df.Country
gdppc_df.index = gdppc_df.Country
pop_df.index = pop_df.Country
co2_df = co2_df.iloc[:,1:]
gdp_df = gdp_df.iloc[:,1:]
gdppc_df = gdppc_df.iloc[:,1:]
pop_df = pop_df.iloc[:,1:]

noworld_co2_df = co2_df.drop("World")
noworld_gdp_df = gdp_df.drop("World")
noworld_gdppc_df = gdppc_df.drop("World")
noworld_pop_df = pop_df.drop("World")





# # # Plots data from all countries for a selected year # # #

x = "2014" # YEAR TO PLOT #


i = 0
k= 0
colors = []
RGBcolors = []

while i < len(noworld_co2_df.index):
    RGB = (np.random.rand(), np.random.rand(), np.random.rand()) 
    colors.append(RGB)
    
    RGB1, RGB2, RGB3 = RGB
    RGB_2 = (RGB1*255, RGB2*255, RGB3*255)
    RGBcolors.append(RGB_2)
    
    i += 1

# Color codes the points based on country. Country color can be determined through a legend of RGB values for the points. 
# This type of color coding/legend is not ideal but it was one of the only ways I was able to differentiate between individual countries.

legend_df = pd.DataFrame(RGBcolors, index=noworld_co2_df.index, columns=['R', 'G', 'B'])


fig = plt.figure(figsize=(9,4))
plt.scatter(noworld_gdp_df.loc[:,x], noworld_co2_df.loc[:,x], c=colors)
plt.xlim(0, 6e12)
plt.grid(axis='both')
plt.title('CO2 Emissions vs GDP')
plt.xlabel('GDP')
plt.ylabel('CO2 Emissions')



fig = plt.figure(figsize=(9,4))
plt.scatter(noworld_gdppc_df.loc[:,x], noworld_co2_df.loc[:,x], c=colors)
plt.grid(axis='both')
plt.title('CO2 Emissions vs GDP per capita')
plt.xlabel('GDP per capita')
plt.ylabel('CO2 Emissions')



fig = plt.figure(figsize=(9,4))
plt.scatter(noworld_pop_df.loc[:,x], noworld_co2_df.loc[:,x], c=colors)
plt.xlim(0, 4e8)
plt.grid(axis='both')
plt.title('CO2 Emissions vs Population')
plt.xlabel('Population')
plt.ylabel('CO2 Emissions')


print(legend_df)





# # # Fit data # # #

fit_gdp_df = noworld_gdp_df[noworld_gdp_df[x] < 6e12] # Removes outlier points
fit_pop_df = noworld_pop_df[noworld_pop_df[x] < 4e8]
fit_gdp_df = fit_gdp_df.loc[:,x]
fit_pop_df = fit_pop_df.loc[:,x]

fitgdp_co2_df = noworld_co2_df[noworld_co2_df.index.isin(fit_gdp_df.index)]
fitpop_co2_df = noworld_co2_df[noworld_co2_df.index.isin(fit_pop_df.index)]
fitgdp_co2_df = fitgdp_co2_df.loc[:,x]
fitpop_co2_df = fitpop_co2_df.loc[:,x]



def gdp_lin(x, a, b):
    return a * x + b

def pop_exp(x, a, b, c):
    return a ** (b * x) + c



popt_gdp, pcov_gdp = curve_fit(gdp_lin, fit_gdp_df, fitgdp_co2_df, p0 = [.000001, 10])
gdp_fucn_vals = gdp_lin(fit_gdp_df, popt_gdp[0], popt_gdp[1])



fig = plt.figure(figsize=(9,4))
plt.scatter(fit_gdp_df, gdp_fucn_vals, c = 'm', label='Fit function')
plt.scatter(fit_gdp_df, fitgdp_co2_df, c='g', alpha=.6, label='Raw data')
plt.grid(axis='both')
plt.title('CO2 Emissions vs GDP (World)')
plt.xlabel('GDP')
plt.ylabel('CO2 Emissions')
plt.legend(loc='lower right')
#print(popt_gdp, pcov_gdp)

"""


popt_gdppc, pcov_gdppc = curve_fit(co2_linear, worldgdppc_df, worldco2_df, 
                         p0 = [1, 10])
gdppc_fucn_vals = co2_linear(worldgdppc_df, popt_gdppc[0], popt_gdppc[1])

fig = plt.figure(figsize=(9,4))
plt.scatter(worldgdppc_df, gdppc_fucn_vals, c = 'm', label='Fit function')
plt.scatter(worldgdppc_df, worldco2_df, c='r', alpha=.6, label='Raw data')
plt.grid(axis='both')
plt.title('CO2 Emissions vs GDP per capita (World)')
plt.xlabel('GDP per capita')
plt.ylabel('CO2 Emissions')
plt.legend(loc='lower right')
#print(popt_gdppc, pcov_gdppc)

"""

# The data for GDP per capita seems to have a considerable amount of holes and does not really show much of a relationship at all.
# Because of this, GDP per capita was no longer analyzed to potentially correlate the most with CO2 emissions.


popt_pop, pcov_pop = curve_fit(pop_exp, fit_pop_df, fitpop_co2_df, 
                         p0 = [1.000001, .1, 10])
pop_fucn_vals = pop_exp(fit_pop_df, popt_pop[0], popt_pop[1], popt_pop[2])


fig = plt.figure(figsize=(9,4))
plt.scatter(fit_pop_df, pop_fucn_vals, c = 'm', linestyle='dashed', label='Fit function')
plt.scatter(fit_pop_df, fitpop_co2_df, c='b', alpha=.6, label='Raw data')
plt.grid(axis='both')
plt.title('CO2 Emissions vs Population (World)')
plt.xlabel('Population')
plt.ylabel('CO2 Emissions')
plt.legend(loc='lower right')
#print(popt_pop, pcov_pop)





# # # Calculate R squared values # # #

residuals_gdp = fitgdp_co2_df - gdp_lin(fit_gdp_df, *popt_gdp)
ss_res_gdp = np.sum(residuals_gdp**2)
ss_tot_gdp = np.sum((fitgdp_co2_df - np.mean(fitgdp_co2_df))**2)
r_squared_gdp = 1 - (ss_res_gdp / ss_tot_gdp)
print("The fit for the GDP data resulted in an R squared value of:", r_squared_gdp)

"""

residuals_gdppc = worldco2_df - co2_linear(worldgdppc_df, *popt_gdppc)
ss_res_gdppc = np.sum(residuals_gdppc**2)
ss_tot_gdppc = np.sum((worldco2_df - np.mean(worldco2_df))**2)
r_squared_gdppc = 1 - (ss_res_gdppc / ss_tot_gdppc)
print(r_squared_gdppc)

"""



residuals_pop = fitpop_co2_df - pop_exp(fit_pop_df, *popt_pop)
ss_res_pop = np.sum(residuals_pop**2)
ss_tot_pop = np.sum((fitpop_co2_df - np.mean(fitpop_co2_df))**2)
r_squared_pop = 1 - (ss_res_pop / ss_tot_pop)
print("The fit for the population data resulted in an R squared value of:", r_squared_pop)



# Looking at the R squared values for both CO2 emissions vs population and GDP, it can be seen that population and CO2 emissions seem to correlate the most.
# Outliers were removed because they were at considerably higher populations and seemed to no longer follow an exponential distrubtion.
# It's not entirely surprising that after a point the exponential relationship looked at at lower populations levels off and takes a new form.
# However, if one is looking at countries with a population under ~330 million, the exponential model seems to provide the best fit.
# It is important to note that the fit itself is obviously not perfect and there is a fair bit of variance from the model at much lower populations and some points
# at mid-sized populations.



# # # Old code that was presented on # # #

"""

# # # Plot data of all countries # # #

fig = plt.figure(figsize=(9,4))
plt.scatter(noworld_gdp_df.iloc[:,1:], noworld_co2_df.iloc[:,1:], c='g')
plt.grid(axis='both')
plt.title('CO2 Emissions vs GDP (All Countries)')
plt.xlabel('GDP')
plt.ylabel('CO2 Emissions')



fig = plt.figure(figsize=(9,4))
plt.scatter(noworld_gdppc_df.iloc[:,1:], noworld_co2_df.iloc[:,1:], c='r')
plt.grid(axis='both')
plt.title('CO2 Emissions vs GDP per capita (All Countries)')
plt.xlabel('GDP per capita')
plt.ylabel('CO2 Emissions')



fig = plt.figure(figsize=(9,4))
plt.scatter(noworld_pop_df.iloc[:,1:], noworld_co2_df.iloc[:,1:], c='b')
plt.grid(axis='both')
plt.title('CO2 Emissions vs Population (All Countries)')
plt.xlabel('Population')
plt.ylabel('CO2 Emissions')





# # # Plots data from just the top economies # # #

topco2_df = co2_df.loc[["United States", "China", "Japan", "Germany", "United Kingdom"]]
topgdp_df = gdp_df.loc[["United States", "China", "Japan", "Germany", "United Kingdom"]]
topgdppc_df = gdppc_df.loc[["United States", "China", "Japan", "Germany", "United Kingdom"]]
toppop_df = pop_df.loc[["United States", "China", "Japan", "Germany", "United Kingdom"]]



fig = plt.figure(figsize=(9,4))
plt.scatter(topgdp_df, topco2_df, c='g')
plt.grid(axis='both')
plt.title('CO2 Emissions vs GDP (Top Economies)')
plt.xlabel('GDP')
plt.ylabel('CO2 Emissions')



fig = plt.figure(figsize=(9,4))
plt.scatter(topgdppc_df, topco2_df, c='r')
plt.grid(axis='both')
plt.title('CO2 Emissions vs GDP per capita (Top Economies)')
plt.xlabel('GDP per capita')
plt.ylabel('CO2 Emissions')



fig = plt.figure(figsize=(9,4))
plt.scatter(toppop_df, topco2_df, c='b')
plt.grid(axis='both')
plt.title('CO2 Emissions vs Population (Top Economies)')
plt.xlabel('Population')
plt.ylabel('CO2 Emissions')

"""



"""

# # # Plots data from individaul countries # # #

y = "Mexico" # INSERT COUNTRY TO PLOT #

singleco2_df = co2_df.loc[y]
singlegdp_df = gdp_df.loc[y]
singlegdppc_df = gdppc_df.loc[y]
singlepop_df = pop_df.loc[y]



fig = plt.figure(figsize=(9,4))
plt.scatter(singlegdp_df, singleco2_df, c='g')
plt.grid(axis='both')
plt.title('CO2 Emissions vs GDP (Single Country)')
plt.xlabel('GDP')
plt.ylabel('CO2 Emissions')



fig = plt.figure(figsize=(9,4))
plt.scatter(singlegdppc_df, singleco2_df, c='r')
plt.grid(axis='both')
plt.title('CO2 Emissions vs GDP per capita (Single Country)')
plt.xlabel('GDP per capita')
plt.ylabel('CO2 Emissions')



fig = plt.figure(figsize=(9,4))
plt.scatter(singlepop_df, singleco2_df, c='b')
plt.grid(axis='both')
plt.title('CO2 Emissions vs Population (Single Country)')
plt.xlabel('Population')
plt.ylabel('CO2 Emissions')





# # # Plot for world # # #

z = "World"

worldco2_df = co2_df.loc[z]
worldgdp_df = gdp_df.loc[z]
worldgdppc_df = gdppc_df.loc[z]
worldpop_df = pop_df.loc[z]



fig = plt.figure(figsize=(9,4))
plt.scatter(worldgdp_df, worldco2_df, c='g')
plt.grid(axis='both')
plt.title('CO2 Emissions vs GDP (World 1990-2018)')
plt.xlabel('GDP')
plt.ylabel('CO2 Emissions')



fig = plt.figure(figsize=(9,4))
plt.scatter(worldgdppc_df, worldco2_df, c='r')
plt.grid(axis='both')
plt.title('CO2 Emissions vs GDP per capita (World 1990-2018)')
plt.xlabel('GDP per capita')
plt.ylabel('CO2 Emissions')



fig = plt.figure(figsize=(9,4))
plt.scatter(worldpop_df, worldco2_df, c='b')
plt.grid(axis='both')
plt.title('CO2 Emissions vs Population (World 1990-2018)')
plt.xlabel('Population')
plt.ylabel('CO2 Emissions')


"""
