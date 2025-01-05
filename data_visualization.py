import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from adjustText import adjust_text

# Title and Introduction
st.title("Data Visualization Dashboard")

#upload files
st.sidebar.header("Upload Your Datasets")
uploaded_file_breakfast = st.sidebar.file_uploader("Upload Breakfast Dataset (CSV/Excel)", type=["csv", "xlsx"], key="breakfast")
uploaded_file_lunch = st.sidebar.file_uploader("Upload Lunch Dataset (CSV/Excel)", type=["csv", "xlsx"], key="lunch")
uploaded_file_dinner = st.sidebar.file_uploader("Upload Dinner Dataset (CSV/Excel)", type=["csv", "xlsx"], key="dinner")


# Helper function to load data
def load_data(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
    return None

# Load datasets
breakfast_data = load_data(uploaded_file_breakfast)
lunch_data = load_data(uploaded_file_lunch)
dinner_data = load_data(uploaded_file_dinner)

# Section Tabs
if any([breakfast_data is not None, lunch_data is not None, dinner_data is not None]):
    tabs = st.tabs(["Graphical Analysis", "Detailed Analysis","Combination"])

    with tabs[0]:
        st.sidebar.header("Select Meal Type")
        meal_type = st.sidebar.selectbox("Choose a meal type:", ["Breakfast", "Lunch", "Dinner"])

        # Helper function to create bar charts
        def create_bar_chart(data, column, title, xlabel, ylabel):
            if column in data.columns:
                plt.figure(figsize=(12, 8))

                # Get the value counts and plot the bar chart
                value_counts = data[column].value_counts().sort_values(ascending=True)
                value_counts.plot(kind='barh', color=sns.color_palette('Dark2'))
                
                # Add title
                plt.title(title)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.gca().spines[['top', 'right']].set_visible(False)

                # Add the count numbers on top of each bar
                for index, value in enumerate(value_counts):
                    plt.text(value + 0.1, index, str(value), va='center', fontweight='bold')
                st.pyplot(plt)
            else:
                st.warning(f"Column '{column}' not found in the dataset.")

        # Meal-specific visualizations
        if meal_type == "Breakfast" and breakfast_data is not None:
            st.subheader("Breakfast Analysis")
            create_bar_chart(breakfast_data, 'Fruit Type', "Fruit Type Distribution", "Count", "Fruit Type")
            create_bar_chart(breakfast_data, 'Chutney Type', "Chutney Type Distribution", "Count", "Chutney Type")
            # Combined Dish Analysis
            st.subheader("Combined Dish Analysis")
            breakfast_data['Dish-1'] = breakfast_data['Dish-1'].str.strip()
            breakfast_data['Dish-2'] = breakfast_data['Dish-2'].str.strip()
            combined_dishes = pd.concat([breakfast_data['Dish-1'], breakfast_data['Dish-2']]).dropna()

            # Count the frequency of each unique dish
            dish_counts = combined_dishes.value_counts()
            top_10_dishes = dish_counts.head(10)

            plt.figure(figsize=(12, 8))
            top_10_dishes.plot(kind='barh', color=sns.color_palette('Dark2'))
            plt.title("Top 10 Most Frequent Combinations of Dish-1 or Dish-2")
            plt.xlabel("Frequency")
            plt.ylabel("Dish Combination")
            #plt.xticks(rotation=45, ha='right')
            plt.gca().spines[['top', 'right']].set_visible(False)

            for index, value in enumerate(top_10_dishes):
                plt.text(value + 0.1, index, str(value), va='center', fontweight='bold')
            plt.tight_layout()
            st.pyplot(plt)
            #plt.show()
            

        elif meal_type == "Lunch" and lunch_data is not None:
            st.subheader("Lunch Analysis")

            lunch_tabs = st.tabs(["Lunch Analysis", "Combination"])
            with lunch_tabs[0]:
                st.write("Analysis of Lunch data")
                create_bar_chart(lunch_data, 'Dal', "Most Occurring Dal", "Frequency", "Dal")
                create_bar_chart(lunch_data, 'Rice', "Most Occurring Rice", "Frequency", "Rice")
                create_bar_chart(lunch_data, 'Bread', "Most Occurring Breads", "Frequency", "Bread")
                create_bar_chart(lunch_data, 'Salad', "Salad Type Distribution", "Salad Type", "Count")

                st.subheader("Combined Dish Analysis")
                lunch_data['Dish 1'] = lunch_data['Dish 1'].str.strip()
                lunch_data['Dish 2'] = lunch_data['Dish 2'].str.strip()
                combined_dishes = pd.concat([lunch_data['Dish 1'], lunch_data['Dish 2']]).dropna()

                # Count the frequency of each unique dish
                dish_counts = combined_dishes.value_counts()
                top_10_dishes = dish_counts.head(10)

                plt.figure(figsize=(12, 8))
                top_10_dishes.plot(kind='barh', color=sns.color_palette('Dark2'))
                plt.title("Top 10 Most Frequent Combinations of Dish-1 or Dish-2")
                plt.xlabel("Frequency")
                plt.ylabel("Dish Combination")
                plt.gca().spines[['top', 'right']].set_visible(False)

                for index, value in enumerate(top_10_dishes):
                    plt.text(value + 0.1, index, str(value), va='center', fontweight='bold')
                st.pyplot(plt)

            with lunch_tabs[1]:
                st.write("Combination of dishes with Dish Category")
                dish_cluster = {'Paneer Dishes': ['Paneer Butter Masala','Matar Paneer','Paneer Capsicum','Paneer Khurchan','Paneer Do Pyaza'],
                        'Aloo Dishes': ['Aloo Posto','Aloo Matar','Dum Aloo Lobiya','Aloo Jeera','Jeera Aloo','Dum Aloo','Dum Aloo Banarasi',
                        'Aloo Mater Ki Sabzi','Aloo Chana Curry','Cabbage Aloo Patta Dry','Aloo Palak','Aloo Capsicum','Aloo Methi','Aloo Rasile']}
                categories = ['Paneer Dishes', 'Aloo Dishes']
                column_names = ['Category', 'Dal', 'Rice', 'Salad', 'Desert', 'Chillies']

                # Define the columns to iterate over
                dish_columns = ['Dish 1', 'Dish 2', 'Dal', 'Rice', 'Salad', 'Desert', 'Chillies']

                def summarize_dishes(dish_cluster, lunch):
                    # Initialize an empty list to store each row of the final table
                    summary = []

                    for category in categories:
                        # Get the dishes for the current category from the dish_cluster dictionary
                        dishes_in_category = dish_cluster[category]

                        # Initialize sets to store the unique types for Dal, Rice, Salad, Desert, and Chillies for this category
                        dal_types = set()
                        rice_types = set()
                        salad_types = set()
                        dessert_types = set()
                        chillies_types = set()

                        # Iterate through all rows in the lunch DataFrame
                        for _, row in lunch.iterrows():
                            # Check if any of the dishes in this row match the current category
                            common_dishes = set(dishes_in_category) & set([row['Dish 1'], row['Dish 2']])

                            if common_dishes:  # If there are any dishes in common
                                # Add the respective types to the sets
                                dal_types.add(row['Dal'])
                                rice_types.add(row['Rice'])
                                salad_types.add(row['Salad'])
                                dessert_types.add(row['Desert'])
                                chillies_types.add(row['Chillies'])

                        # Append the category and corresponding types to the summary list
                        summary.append([
                            category,
                            ', '.join(str(dal) for dal in dal_types) if dal_types else 'None',
                            ', '.join(str(rice) for rice in rice_types) if rice_types else 'None',
                            ', '.join(str(salad) for salad in salad_types) if salad_types else 'None',
                            ', '.join(str(dessert) for dessert in dessert_types) if dessert_types else 'None',
                            ', '.join(str(chillies) for chillies in chillies_types) if chillies_types else 'None'
                        ])
                        summary_df = pd.DataFrame(summary, columns=column_names)
                    return summary_df
                
                # Displaying the table interactively
                summary_df = summarize_dishes(dish_cluster, lunch_data)
                st.dataframe(summary_df, use_container_width=True)

                # Allow downloading the combinations table as a CSV
                csv = summary_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Combinations as CSV",
                    data=csv,
                    file_name="common_dishes_combinations.csv",
                    mime='text/csv',
                )

        elif meal_type == "Dinner" and dinner_data is not None:
            st.subheader("Dinner Analysis")
            create_bar_chart(dinner_data, 'Curry', "Most Occurring Curry", "Count", "Main Course")
            create_bar_chart(dinner_data, 'Dal Type', "Most Occurring Dal Types", "Count", "Dal")
            create_bar_chart(dinner_data, 'Type of dish made of rice', "Most Occurring type of dish made of rice", "Count", "Dish made of rice")
            create_bar_chart(dinner_data, 'Salad', "Most Occurring Salad", "Count", "Salad")
            create_bar_chart(dinner_data, 'Soup', "Most Occurring soups", "Count", "Soups")
            create_bar_chart(dinner_data, 'Dessert', "Dessert Distribution", "Count", "Dessert")

            st.subheader("Combined Dish Analysis")
            dinner_data['Dish 1'] = dinner_data['Dish 1'].str.strip()
            dinner_data['Dish 2'] = dinner_data['Dish 2'].str.strip()
            combined_dishes = pd.concat([dinner_data['Dish 1'], dinner_data['Dish 2']]).dropna()

            # Count the frequency of each unique dish
            dish_counts = combined_dishes.value_counts()
            top_10_dishes = dish_counts.head(10)

            plt.figure(figsize=(12, 8))
            top_10_dishes.plot(kind='barh', color=sns.color_palette('Dark2'))
            plt.title("Top 10 Most Frequent Combinations of Dish-1 or Dish-2")
            plt.xlabel("Frequency")
            plt.ylabel("Dish Combination")
            plt.gca().spines[['top', 'right']].set_visible(False)

            for index, value in enumerate(top_10_dishes):
                plt.text(value + 0.1, index, str(value), va='center', fontweight='bold')
            st.pyplot(plt)
        else:
            st.warning(f"Dataset for {meal_type} is not uploaded.")

    with tabs[1]:
        st.subheader("Detailed Analysis")

        # Dropdown to select meal type
        dataset_type = st.selectbox("Select Dataset Type", ["Breakfast", "Lunch", "Dinner"])

        # Dynamically load the dataset based on the selected meal type
        if dataset_type == "Breakfast":
            dataset = breakfast_data
        elif dataset_type == "Lunch":
            dataset = lunch_data
        elif dataset_type == "Dinner":
            dataset = dinner_data
        else:
            dataset = None

        if dataset is not None:
            # Display the relevant dataset preview
            st.write(f"Preview of the {dataset_type} Dataset")
            st.write(dataset.head())

            # Dish keyword input for network graph
            dish_keyword = st.text_input("Enter a Dish Keyword for Network Analysis", "")

            if st.button("Generate Network Graph", key="button1"):
                if dish_keyword.strip():  # Only process if the input is not empty
                    # Dynamically define dish columns based on the dataset type
                    if dataset_type == 'Dinner':
                        dish_columns = [col for col in dataset.columns if col.startswith(('Dish', 'Curd', 'Dessert', 'Dal', 'Soup', 'Chapati', 'Chutney', 'Pickle', 'Lemon wedges', 'Chillies', 'Salad', 'Onions', 'Fryums', 'Curry'))]
                    elif dataset_type == 'Lunch':
                        dish_columns = [col for col in dataset.columns if col.startswith(('Dish', 'Dal', 'Rice', 'Salad', 'Desert', 'Drinks', 'Chillies', 'Pickle', 'Bread'))]
                    elif dataset_type == 'Breakfast':
                        dish_columns = [col for col in dataset.columns if col.startswith(('Dish', 'Bread', 'Cereals', 'Tea', 'Fruit', 'Chutney'))]
                    else:
                        st.error("Invalid dataset type selected.")
                        dish_columns = []

                    if dish_columns:
                        # Filter rows based on the dish keyword
                        relevant_rows = dataset[
                            dataset[dish_columns].apply(lambda row: row.str.contains(dish_keyword, case=False, na=False).any(), axis=1)
                        ]

                        if relevant_rows.empty:
                            st.warning(f"No data found for the keyword '{dish_keyword}'.")
                        else:
                            # Count occurrences of all dishes in the relevant rows
                            all_dishes = relevant_rows[dish_columns].stack()
                            central_frequency = len(relevant_rows)
                            dish_counts = all_dishes[~all_dishes.str.contains(dish_keyword, case=False, na=False)].value_counts()

                            if dish_counts.empty:
                                st.warning(f"No dish combinations found for '{dish_keyword}'.")
                            else:
                                # Create the network graph
                                G = nx.Graph()

                                # Add central node
                                G.add_node(
                                    dish_keyword,
                                    size=1200,
                                    color='red',
                                    label=f"{dish_keyword} (Freq: {central_frequency})"
                                )

                                # Add other nodes and edges
                                for dish, count in dish_counts.items():
                                    G.add_node(
                                        dish,
                                        size=count * 100,
                                        color='skyblue',
                                        label=f"{dish} (Freq: {count})"
                                    )
                                    G.add_edge(dish_keyword, dish, weight=count)

                                # Generate positions
                                pos = nx.spring_layout(G, k=0.5, seed=42)

                                # Node attributes
                                sizes = [G.nodes[node].get('size', 300) for node in G]
                                colors = [G.nodes[node].get('color', 'skyblue') for node in G]
                                labels = nx.get_node_attributes(G, 'label')

                                # Create the plot
                                plt.figure(figsize=(12, 8))
                                plt.axis('off')
                                nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, alpha=0.8)
                                nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, width=1)

                                # Add labels
                                texts = []
                                for node, (x, y) in pos.items():
                                    texts.append(
                                        plt.text(
                                            x, y, labels[node],
                                            fontsize=9,
                                            ha='center',
                                            va='center',
                                            color='black'
                                        )
                                    )
                                adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))

                                plt.title(f"Network of Dishes Made with '{dish_keyword}' ({dataset_type} Dataset)", fontsize=16)
                                st.pyplot(plt)
                    else:
                        st.warning(f"No relevant columns found for {dataset_type} dataset.")
        else:
            st.warning(f"Please upload the {dataset_type} dataset to proceed.")
    
    with tabs[2]:
        st.header("Combination of Lunch and Dinner Dishes")
        st.write("This table displays common dishes between lunch and dinner with respective dates.")
        
        rename_mapping = {
            'Type of dish made of rice': 'Rice',
            'Dal Type': 'Dal',
            'Type of chillies': 'Chillies'
        }

        if dinner_data is not None and lunch_data is not None:
            dinner_data.rename(columns=rename_mapping, inplace=True)
            lunch_combined = lunch_data.melt(id_vars='Date',
                                value_vars=['Dish 1', 'Dish 2', 'Dal', 'Rice', 'Bread', 'Salad', 'Pickle', 'Drinks', 'Desert', 'Chillies'],
                                var_name='Dish_Type',
                                value_name='Dish')

            dinner_combined = dinner_data.melt(id_vars='date',
                            value_vars=['Onions', 'Fryums', 'Chapati', 'Salad', 'Pickle', 'Rice', 'Dal',
                                            'Curd /Alternative', 'Chillies', 'Dessert', 'Lemon wedges',
                                            'Dish 1', 'Dish 2', 'Dish 3', 'Chutney', 'Curry', 'Soup',
                                            'Garlic croutons'],
                            var_name='Dish_Type',
                            value_name='Dish')
        
            # Rename 'date' to 'Date' for consistency
            dinner_combined.rename(columns={'date': 'Date'}, inplace=True)

            # Normalize text
            lunch_combined['Dish'] = lunch_combined['Dish'].str.lower().fillna('')
            dinner_combined['Dish'] = dinner_combined['Dish'].str.lower().fillna('')

            lunch_set = set(lunch_combined['Dish']) - {''}
            dinner_set = set(dinner_combined['Dish']) - {''}

            common_dishes = lunch_set.intersection(dinner_set)

            filtered_lunch = lunch_combined[lunch_combined['Dish'].isin(common_dishes)]
            filtered_dinner = dinner_combined[dinner_combined['Dish'].isin(common_dishes)]

            # Full Outer join to include all common dishes with dates
            common_dishes_with_dates = pd.merge(
                filtered_lunch,
                filtered_dinner,
                on='Dish',
                how='outer',
                suffixes=('_lunch', '_dinner')
            )
            # Convert dates to datetime format
            common_dishes_with_dates['Date_lunch'] = pd.to_datetime(
                            common_dishes_with_dates['Date_lunch'], format='%d-%m-%Y', errors='coerce')
            common_dishes_with_dates['Date_dinner'] = pd.to_datetime(
                            common_dishes_with_dates['Date_dinner'], format='%d-%m-%Y', errors='coerce')

            # Selecting the required columns
            common_dishes_with_dates = common_dishes_with_dates[['Dish', 'Date_lunch', 'Date_dinner']]

            # Group by 'Dish' and aggregate the dates, sorting them internally and formatting the date as 'dd-mm-yyyy'
            result = common_dishes_with_dates.groupby('Dish').agg(
                lunch_dates=('Date_lunch', lambda x: [d.strftime('%d-%m-%Y') for d in sorted(x.unique())]),
                dinner_dates=('Date_dinner', lambda x: [d.strftime('%d-%m-%Y') for d in sorted(x.unique())])
            ).reset_index()

            # Displaying the table interactively
            st.dataframe(result, use_container_width=True)

            # Optionally, allow downloading the table as a CSV
            csv = result.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Table as CSV",
                data=csv,
                file_name="common_dishes.csv",
                mime='text/csv',
            )
        else:
            st.write("Please upload the Lunch and Dinner dataset")