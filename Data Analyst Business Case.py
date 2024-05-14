#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 22:25:32 2024

@author: Harikrishnan Marimuthu
"""
# Importing libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to the dataset files:
dataset1_path = "Task 1 - Dataset 1.xlsx"
dataset2 = pd.read_excel("Task 1 - Dataset 2.xlsx",
                         sheet_name="Digihaul Average Rates")

# Function to clean the dataset1 from an Excel file:
def load_and_clean_data(path, sheet_names):
    data = pd.read_excel(path, sheet_name=sheet_names)
    for week, week_data in data.items():
        week_data.columns = week_data.iloc[2]
        week_data = week_data[3:]
        week_data = week_data.loc[:, week_data.columns.notnull()]
        week_data = week_data.apply(pd.to_numeric, errors='ignore')
        data[week] = week_data
    return data


# Define weeks and the rate columns to analyze:
weeks = ['Apr-21 Week1', 'Apr-21 Week2', 'Apr-21 Week3',
         'Apr-21 Week4', 'May-21 Week1', 'May-21 Week2', 'May-21 Week3']
rate_columns = ['Rate (24h)', 'Rate (48h)', 'Rate (72h)', 'Rate (96h)']

dataset1 = load_and_clean_data(dataset1_path, weeks)

# Calculate and store the average rates for each week:
average_rates_per_week = {}
for week, data in dataset1.items():
    week_averages = {}
    for rate in rate_columns:
        if rate in data.columns:
            week_averages[f'Average {rate}'] = data[rate].mean(axis=1).mean()
    average_rates_per_week[week] = week_averages

rate_summary_df = pd.DataFrame.from_dict(average_rates_per_week).T.round(2)
rate_summary_df.reset_index(inplace=True)
rate_summary_df.rename(columns={'index': 'Week'}, inplace=True)

# Prepare the second dataset for analysis by mapping week numbers to dataset1 weeks:
dataset2_cleaned = dataset2.iloc[:, :3].dropna()
dataset2_cleaned.columns = ['LaneArea', 'WeekNumber', 'Average Rate (48h)']
dataset2_cleaned['Average Rate (48h)'] = dataset2_cleaned['Average Rate (48h)'].round(
    2)
weeks = dataset1.keys()
week_mapping = {i: week for i, week in enumerate(weeks, start=15)}
dataset2_cleaned['Week'] = dataset2_cleaned['WeekNumber'].map(week_mapping)
grouped_mean = dataset2_cleaned.groupby(
    'Week')['Average Rate (48h)'].mean().round(2).reset_index()
grouped_mean.columns = ['Week', 'Average Rate (48h)']

# Function to process and calculate average rates from detailed weekly data:
def finalize_week_data_corrected(week_data):
    rate_24h_cols = [col for col in week_data.columns if 'Rate (24h)' in col]
    rate_48h_cols = [col for col in week_data.columns if 'Rate (48h)' in col]
    rate_72h_cols = [col for col in week_data.columns if 'Rate (72h)' in col]
    rate_96h_cols = [col for col in week_data.columns if 'Rate (96h)' in col]

    week_data['Average Rate (24h)'] = week_data[rate_24h_cols].astype(
        float).mean(axis=1)
    week_data['Average Rate (48h)'] = week_data[rate_48h_cols].astype(
        float).mean(axis=1)
    week_data['Average Rate (72h)'] = week_data[rate_72h_cols].astype(
        float).mean(axis=1)
    week_data['Average Rate (96h)'] = week_data[rate_96h_cols].astype(
        float).mean(axis=1)
    total_loads = week_data['Total Loads'].astype(int).mean(axis=1)

    final_df = pd.DataFrame({
        'Lane (Area)': week_data['Lane (Area)'].iloc[:, 0],
        'Total Loads': total_loads,
        'Average Rate (24h)': week_data['Average Rate (24h)'],
        'Average Rate (48h)': week_data['Average Rate (48h)'],
        'Average Rate (72h)': week_data['Average Rate (72h)'],
        'Average Rate (96h)': week_data['Average Rate (96h)']
    })
    final_df.reset_index(drop=True, inplace=True)
    return final_df


# Applying the finalization function to each week's dataset:
corrected_final_weeks_data = {week: finalize_week_data_corrected(
    dataset1[week]) for week in dataset1}

# Combining data from all weeks into a single DataFrame.
all_weeks_data = pd.concat(
    [data for data in corrected_final_weeks_data.values()])

# Calculating mean rates for 48-hour bookings by lane:
avg_48h_by_lane = all_weeks_data.groupby(
    'Lane (Area)')['Average Rate (48h)'].mean()

# Identifying top 5 lanes with the highest average 48h rates:
top_5_lanes_avg_48h = avg_48h_by_lane.sort_values(ascending=False).head(5)
top_5_lanes_avg_48h_df = top_5_lanes_avg_48h.reset_index()
top_5_lanes_avg_48h_df.columns = ['Lane (Area)', 'Average Rate (48h)']

# Calculating an overall average rate from all durations across all weeks:
all_weeks_data['Average Rate'] = all_weeks_data[[
    'Average Rate (24h)', 'Average Rate (48h)', 'Average Rate (72h)', 'Average Rate (96h)']].mean(axis=1)

# Extracting final summary data for lanes, loads, and average rates:
final_df = all_weeks_data[[
    'Lane (Area)', 'Total Loads', 'Average Rate']].copy()

# Filtering data for DN postcodes and calculating average rates:
dn_postcodes_data = final_df[final_df['Lane (Area)'].str.startswith('DN')]
average_rates_by_postcode = dn_postcodes_data.groupby('Lane (Area)')[
    'Average Rate'].mean()

# Identifying the top and least five postcodes based on average rates:
top_five_postcodes = average_rates_by_postcode.nlargest(5)
least_five_postcodes = average_rates_by_postcode.nsmallest(5)

# Calculating average rate per load to see the cost efficiency:
final_df['Average Rate per Load'] = final_df['Average Rate'] / \
    final_df['Total Loads']

# Summarizing load and rate data by lane area:
lane_summary = final_df.groupby('Lane (Area)').agg({
    'Total Loads': 'mean',
    'Average Rate per Load': 'mean'
}).reset_index()

# Removing duplicates in lane summary:
lane_summary = lane_summary.drop_duplicates(subset=['Lane (Area)'])

# Plotting a pie chart of the top 5 lane areas by average 48-hour rates:
lane_area_means = dataset2_cleaned.groupby(
    'LaneArea')['Average Rate (48h)'].mean().sort_values().round(2).reset_index()
top_5_lane_areas = lane_area_means.nlargest(5, 'Average Rate (48h)')
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
explode = (0.1, 0, 0, 0, 0)
plt.figure(figsize=(10, 5))
plt.pie(top_5_lane_areas['Average Rate (48h)'], labels=top_5_lane_areas['LaneArea'], autopct='%1.1f%%', startangle=140,
        explode=explode, colors=colors, shadow=True)
plt.title('Top 5 Lane Areas by Average Rate for Digihaul (48h)',
          fontsize=14, fontweight='bold')
plt.axis('equal')
plt.legend(fontsize=10, loc='best', frameon=True, shadow=True, fancybox=True)
plt.tight_layout()
plt.show()

# Plotting donut charts showing the percentage distribution of origin postcodes across different weeks:
origin_postcodes = pd.concat([week_data['Lane (Area)'].str.split(
    '-', expand=True)[0] for week, week_data in corrected_final_weeks_data.items()])
postcode_counts = origin_postcodes.value_counts()
total_count = postcode_counts.sum()
percentage = (postcode_counts / total_count) * 100
colors = sns.light_palette("teal", n_colors=2)[:2]
colors = [colors[1], colors[0]]
plt.figure(figsize=(14, 8))
num_postcodes = len(postcode_counts)
for i, (postcode, percent) in enumerate(percentage.items(), start=1):
    ax = plt.subplot(2, (num_postcodes + 1) // 2, i)
    ax.pie([percent, 100 - percent], colors=colors, startangle=90,
           counterclock=False, wedgeprops=dict(width=0.4, edgecolor='w'))
    ax.set_title(f'Postcode: {postcode}\n{percent:.2f}%',
                 fontsize=12, fontweight='bold', color='navy')
    ax.axis('equal')
plt.suptitle('Distribution of Origin Postcode',
             fontsize=14, fontweight='bold', color='black')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Display scatter plots for the top and least five postcodes by average rate:
colors = ['green', 'red']
categories = ['Top', 'Least']

top_five_postcodes_df = pd.DataFrame(top_five_postcodes).reset_index()
top_five_postcodes_df['Category'] = 'Top'
least_five_postcodes_df = pd.DataFrame(least_five_postcodes).reset_index()
least_five_postcodes_df['Category'] = 'Least'

all_postcodes_df = pd.concat([top_five_postcodes_df, least_five_postcodes_df])

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

for ax, category, color in zip(axes, categories, colors):
    subset = all_postcodes_df[all_postcodes_df['Category'].str.contains(
        category)]
    ax.scatter(subset['Lane (Area)'], subset['Average Rate'],
               color=color, label=f'{category} Five', s=100)

    for i, row in subset.iterrows():
        ax.annotate(f"{row['Average Rate']:.2f}",
                    (row['Lane (Area)'], row['Average Rate']),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center')
    ax.set_title(f'{category} Five DN Postcode Ranked by Average Rate',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Postcode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Rate', fontsize=12, fontweight='bold')
    ax.set_ylim(250, 650)
    ax.set_yticks(np.arange(250, 701, 50))
    ax.tick_params(axis='x', labelsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.legend()
plt.tight_layout()
plt.show()

# Plotting a scatter graph to analyze how the average rate per load varies with the total number of loads:
plt.figure(figsize=(12, 6))
scatter = plt.scatter(lane_summary['Total Loads'], lane_summary['Average Rate per Load'],
                      c=lane_summary['Average Rate per Load'], cmap='inferno', marker='h')

plt.title('Rate Variation over Loads', fontsize=14, fontweight='bold')
plt.xlabel('Loads', fontsize=12, fontweight='bold')
plt.ylabel('Average Rate per Load', fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--')
plt.show()

# Visualize the highest average rates for different time durations across top 5 lanes:
combined_data = pd.concat(corrected_final_weeks_data.values())

avg_rate_per_lane = combined_data.groupby('Lane (Area)').mean()

top_5_lanes = pd.DataFrame({
    'Average Rate (24h)': avg_rate_per_lane['Average Rate (24h)'].nlargest(5),
    'Average Rate (48h)': avg_rate_per_lane['Average Rate (48h)'].nlargest(5),
    'Average Rate (72h)': avg_rate_per_lane['Average Rate (72h)'].nlargest(5),
    'Average Rate (96h)': avg_rate_per_lane['Average Rate (96h)'].nlargest(5)
})

top_5_lanes = top_5_lanes.transpose()

colors = ['teal', 'darkorange', '#F0E68C', '#F08080', '#C0A9C0']

fig, ax = plt.subplots(figsize=(12, 8))
width = 0.15
n = len(top_5_lanes.index)

for i, column in enumerate(top_5_lanes.columns):
    x_offset = np.arange(n) + i * width
    bars = ax.bar(x_offset, top_5_lanes[column],
                  width=width, label=column, color=colors[i])
    # Add text annotations on the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2),
                va='bottom', ha='center', color='black', fontsize=8)
plt.title('Top 5 Lanes with the Highest Average Rates for Each Rate Type',
          fontsize=14, fontweight='bold')
plt.xlabel('Time', fontsize=12, fontweight='bold')
plt.ylabel('Average Rate', fontsize=12, fontweight='bold')
plt.xticks(np.arange(n) + width, top_5_lanes.index)
plt.legend(title='Lanes', loc='best', frameon=True, shadow=True, fancybox=True)
plt.tight_layout()
plt.show()

# Plot the average rates over weeks for different rate durations:
line_styles = ['-', '--', '-.', ':']
colors = ['purple', 'red', 'green', 'darkorange']

plt.figure(figsize=(12, 6))

for i, rate in enumerate(rate_columns):
    if f'Average {rate}' in rate_summary_df:
        plt.plot(rate_summary_df['Week'], rate_summary_df[f'Average {rate}'],
                 marker='o', linestyle=line_styles[i % len(line_styles)],
                 color=colors[i % len(colors)],
                 label=f'Average {rate}')

plt.title('Average Rate Over Weeks', fontsize=14, fontweight='bold')
plt.xlabel('Week', fontsize=12, fontweight='bold')
plt.ylabel('Average Rate', fontsize=12, fontweight='bold')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Rate Types', fontsize=10, loc='best',
           frameon=True, shadow=True, fancybox=True)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Compare and annotate weekly 48-hour average rates between Digihaul and a competitor:
plt.figure(figsize=(12, 6))

line_digihaul, = plt.plot(grouped_mean['Week'], grouped_mean['Average Rate (48h)'],
                          marker='x', linestyle='--', color='darkorange', label='Digihaul Rates')
line_competitor, = plt.plot(
    rate_summary_df['Week'], rate_summary_df['Average Rate (48h)'], marker='o', color='teal', label='Competitor Rates')

for i, txt in enumerate(grouped_mean['Average Rate (48h)']):
    plt.annotate(f"{txt}", (grouped_mean['Week'][i], grouped_mean['Average Rate (48h)'][i]),
                 textcoords="offset points", xytext=(0, 10), ha='center', color='black')

for i, txt in enumerate(rate_summary_df['Average Rate (48h)']):
    plt.annotate(f"{txt}", (rate_summary_df['Week'][i], rate_summary_df['Average Rate (48h)']
                 [i]), textcoords="offset points", xytext=(0, -10), ha='center', color='black')

plt.title('Weekly Comparison of Digihaul and Competitor Rates Over 48h Lead Time',
          fontsize=14, fontweight='bold')
plt.xlabel('Week', fontsize=12, fontweight='bold')
plt.ylabel('Average Rate (48h)', fontsize=12, fontweight='bold')
plt.xticks(fontsize=10)
plt.legend(handles=[line_digihaul, line_competitor], fontsize=10,
           loc='best', frameon=True, shadow=True, fancybox=True)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Plotting the percentage difference in 48-hour rates between Digihaul and a competitor across weeks:
percentage_difference = pd.merge(rate_summary_df[[
                                 'Week', 'Average Rate (48h)']], grouped_mean, on='Week', suffixes=('_Comp', '_Digi'))
percentage_difference['Percentage Difference'] = ((percentage_difference['Average Rate (48h)_Digi'] -
                                                  percentage_difference['Average Rate (48h)_Comp']) / percentage_difference['Average Rate (48h)_Comp']) * 100

plt.figure(figsize=(12, 6))
plt.fill_between(
    percentage_difference['Week'], percentage_difference['Percentage Difference'], color='skyblue', alpha=0.5)
plt.plot(percentage_difference['Week'], percentage_difference['Percentage Difference'],
         color='skyblue', alpha=0.8, linewidth=2)
plt.title('Percentage Difference Between Digihaul and Competitor 48h Rates Over Time',
          fontsize=14, fontweight='bold')
plt.xlabel('Week', fontsize=12, fontweight='bold')
plt.ylabel('Percentage Difference (%)', fontsize=12, fontweight='bold')
plt.xticks(fontsize=10)
plt.axhline(0, color='red', linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualize and compare average 48-hour rates for top lane areas between Digihaul and a competitor:
lanes = top_5_lanes_avg_48h_df['Lane (Area)'].values
rates = top_5_lanes_avg_48h_df['Average Rate (48h)'].values
lane_areas = top_5_lane_areas['LaneArea'].values
area_rates = top_5_lane_areas['Average Rate (48h)'].values

fig, ax = plt.subplots(figsize=(12, 6))

height = 0.35

bar_positions_digihaul = np.arange(len(lane_areas))
bar_positions_competitor = bar_positions_digihaul + height + 0.1

bars1 = ax.barh(bar_positions_digihaul, area_rates, height,
                color='darkorange', label='Digihaul')
bars2 = ax.barh(bar_positions_competitor, rates, height,
                color='teal', label='Competitor')

for bars, labels in zip([bars1, bars2], [lane_areas, lanes]):
    for bar, label in zip(bars, labels):
        width = bar.get_width()
        ax.annotate(label,
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0),
                    textcoords="offset points",
                    ha='left', va='center')

ax.set_ylabel('Area', fontsize=12, fontweight='bold')
ax.set_xlabel('Rate', fontsize=12, fontweight='bold')
ax.set_title('Variation of Rate over Region', fontsize=14, fontweight='bold')
ax.set_yticks([])
ax.legend(fontsize=10, loc='best', frameon=True, shadow=True, fancybox=True)
plt.tight_layout()
plt.show()
