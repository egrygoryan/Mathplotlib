import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('cleaned_airbnb_data.csv')
def classify_availability(availability):
    if availability < 50:
        return "Rarely Available"
    elif availability <= 200:
        return "Occasionally Available"
    else:
        return "Highly Available"
df['availability_status'] = df['availability_365'].apply(classify_availability)


def neighborhood_distribution(df):
    plt.figure(figsize=(10, 7))
    
    bar_colors = ['tab:green', 'tab:blue', 'tab:red', 'tab:orange', 'tab:brown']
    neighborhood_counts = df['neighbourhood_group'].value_counts() 
    
    bars = plt.bar(neighborhood_counts.index, neighborhood_counts.values, color=bar_colors)
    
    plt.xlabel('Neighborhood Group')
    plt.ylabel('Number of Listings')
    plt.title('Neighborhood Distribution')

    for bar in bars:
            bar_height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, bar_height, f'{bar_height}', ha='center', va='bottom')

    plt.savefig('neighborhood_distribution.png')    
    plt.show()

def price_distribution(df):
    plt.figure(figsize =(10, 7))

    neighborhood_groups = df['neighbourhood_group'].unique()
    data = [df[df['neighbourhood_group'] == group]['price'] for group in neighborhood_groups]
    
    #create a boxplot
    bp = plt.boxplot(data, labels=neighborhood_groups, patch_artist = True,
                notch ='True', vert = 1)
    
    colors = ['tab:green', 'tab:blue', 'tab:red', 'tab:orange', 'tab:brown']

    # apply different colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.title('Price Distribution Across Neighborhoods')
    plt.xlabel('Neighborhood Group')
    plt.ylabel('Price')
    plt.ylim(0, 500)

    plt.savefig('price_distribution.png')
    plt.show()

def room_type_vs_availability(df):
    avg_availability = df.groupby(['neighbourhood_group', 'room_type'])['availability_365'].mean().unstack()
    std_availability = df.groupby(['neighbourhood_group', 'room_type'])['availability_365'].std().unstack()
    
    avg_availability.plot(kind='bar', yerr=std_availability, capsize=4, figsize=(12, 8))
    plt.title('Room Type vs. Availability')
    plt.xlabel('Neighborhood Group')
    plt.ylabel('Average Availability')
    plt.legend(title='Room Type')
    plt.xticks(rotation=0)
    plt.savefig('room_type_vs_availability.png')

    plt.show()

def price_vs_reviews(df):
    plt.figure(figsize=(10, 7))
    room_types = df['room_type'].unique()
    
    for room_type in room_types:
        subset_per_room = df[df['room_type'] == room_type]
        plt.scatter(x=subset_per_room['price'], y=subset_per_room['number_of_reviews'], label=room_type)
    
    plt.title('Correlation Between Price and Number of Reviews')
    plt.xlabel('Price')
    plt.ylabel('Number of Reviews')
    
    # regression line, using numpy
    coeffs = np.polyfit(df['price'], df['number_of_reviews'], 1)
    poly = np.poly1d(coeffs)

    plt.plot(df['price'], poly(df['price']), color='red', linestyle='--', linewidth=1)
    
    plt.legend(title='Room Type')
    plt.savefig('price_vs_reviews.png')

    plt.show()

def reviews_over_time(df):
    df['last_review'] = pd.to_datetime(df['last_review'])
    df['year'] = df['last_review'].dt.to_period('Y')

    reviews_trend = df.groupby(['neighbourhood_group', 'year'])['number_of_reviews'].count().unstack()
    reviews_trend = reviews_trend.rolling(window=300, min_periods=1).mean()

    plt.figure(figsize=(12, 8))
    for neighborhood in reviews_trend.index:
        plt.plot(reviews_trend.columns.to_timestamp(), reviews_trend.loc[neighborhood], label=neighborhood)

    plt.title('Time series analysis')
    plt.xlabel('Date')
    plt.ylabel('Number of reviews')
    plt.legend(title='Neighborhood')
    plt.xticks(rotation=45)
    plt.savefig('reviews_over_time.png')

    plt.show()

def price_availability_heatmap(df):
    pivot= df.pivot_table(values='price', index='neighbourhood_group', columns='availability_status', aggfunc='mean')
    plt.figure(figsize=(12, 8))
    setttings = plt.pcolormesh(pivot.columns, pivot.index, pivot, cmap='cividis', shading='nearest')
    plt.colorbar(setttings, label='Average price')
    plt.title('Price and availability heatmap')
    plt.xlabel('Availability status')
    plt.ylabel('Neighborhood group')
    plt.savefig('price_availability_heatmap.png')
    plt.show()

def room_type_review_count(df):
    review_counts = df.groupby(['neighbourhood_group', 'room_type'])['number_of_reviews'].count().unstack()
    review_counts.plot(kind='bar', stacked=True, figsize=(12, 8))
    plt.title('Room type and review count analysis')
    plt.xlabel('Neighborhood')
    plt.ylabel('Total number of reviews')
    plt.xticks(rotation=0)
    plt.savefig('room_type_review_count.png')

    plt.show()


def main():
    # call all graphs
    neighborhood_distribution(df)
    price_distribution(df)
    room_type_vs_availability(df)
    price_vs_reviews(df)
    reviews_over_time(df)
    price_availability_heatmap(df)
    room_type_review_count(df)


if __name__ == '__main__':
    main()