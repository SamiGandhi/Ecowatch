import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # Import for custom legend patches

# Load the Excel file
file_path = 'C:\\Users\\Gandhi\\Desktop\\Annaba Climate.xlsx'  
data = pd.read_excel(file_path)

humidity_data = data[data['Unnamed: 0'] == 'Humidity (%)'].drop('Unnamed: 0', axis=1).iloc[0]
ndvi_mean_data = data[data['Unnamed: 0'] == 'NDVI mean'].drop('Unnamed: 0', axis=1).iloc[0]
months = humidity_data.index

season_colors = {'January': 'red', 'February': 'red', 'March': 'green', 'April': 'green', 'May': 'green',
                 'June': 'blue', 'July': 'blue', 'August': 'blue', 'September': 'orange', 'October': 'orange',
                 'November': 'orange', 'December': 'red'}


# Plotting
fig, ax1 = plt.subplots(figsize=(5, 5))

# Bar plot for humidity
ax1.set_xlabel('Month',fontsize=8)
ax1.set_ylabel('Humidity (%)', color='tab:blue',fontsize=8)
bars = ax1.bar(months, humidity_data, linewidth=1, width=0.5, label='Humidity (%)',
               color=[season_colors[month] for month in months],edgecolor='black')
for bar in bars.patches:
    height = bar.get_height()
    ax1.axhline(y=height, color='tab:gray', linestyle='--', linewidth=0.5)
ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=8, width = 3)
ax1.tick_params(axis='x', labelsize=8, rotation=90, width = 3)
ax1.set_ylim(0, 109)
# Line plot for mean NDVI on a secondary y-axis
ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Mean NDVI', color=color, fontsize=8)
lines = ax2.plot(months, ndvi_mean_data, color=color, marker='.', linestyle='-', label='Mean NDVI',linewidth =1)
for month, ndvi_value in zip(months, ndvi_mean_data):
    print('redline')
    ax2.axhline(y=ndvi_value, color=color, linestyle='--', linewidth=0.5)
ax2.tick_params(axis='y', labelcolor=color, labelsize=9, width = 3)
ax2.set_ylim(-1, 1.09)

season_patches = [mpatches.Patch(color='red', label='Winter (Humidity 76.33 %) (NDVI = 0.54)'),
                  mpatches.Patch(color='green', label='Spring (Humidity 72.00 %) (NDVI = 0.54)'),
                  mpatches.Patch(color='blue', label='Summer (Humidity 57.00 %) (NDVI = 0.37)'),
                  mpatches.Patch(color='orange', label='Autumn (Humidity 69.00 %) (NDVI = 0.32)')]


# Add the legend to the plot
legend1 = ax1.legend(handles=season_patches, loc='upper left', title="Seasons", fontsize='xx-small',title_fontsize='xx-small',edgecolor='black')
# Add the first legend manually to the plot again
ax1.add_artist(legend1)

fig.tight_layout()  # To ensure no overlap of plot elements
plt.show()
