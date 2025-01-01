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
    tab1, tab2, tab3 = st.tabs(["Graphical Analysis", "Detailed Analysis","Relationship b/w different dataset"])

    with tab1:
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
            plt.xlabel("Dish Combination")
            plt.ylabel("Frequency")
            plt.gca().spines[['top', 'right']].set_visible(False)

            for index, value in enumerate(top_10_dishes):
                plt.text(value + 0.1, index, str(value), va='center', fontweight='bold')
            st.pyplot(plt)
            plt.show()
            

        elif meal_type == "Lunch" and lunch_data is not None:
            st.subheader("Lunch Analysis")
            create_bar_chart(lunch_data, 'Dal', "Most Occurring Dal", "Dal", "Frequency")
            create_bar_chart(lunch_data, 'Rice', "Most Occurring Rice", "Rice", "Frequency")
            create_bar_chart(lunch_data, 'Bread', "Most Occurring Breads", "Breads", "Frequency")
            create_bar_chart(lunch_data, 'Salad', "Salad Type Distribution", "Count", "Salad Type")

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
            plt.xlabel("Dish Combination")
            plt.ylabel("Frequency")
            plt.gca().spines[['top', 'right']].set_visible(False)

            for index, value in enumerate(top_10_dishes):
                plt.text(value + 0.1, index, str(value), va='center', fontweight='bold')
            st.pyplot(plt)


        elif meal_type == "Dinner" and dinner_data is not None:
            st.subheader("Dinner Analysis")
            create_bar_chart(dinner_data, 'Curry', "Most Occurring Curry", "Count", "Main Course")
            create_bar_chart(dinner_data, 'Dal Type', "Most Occurring Dal Types", "Count", "Dessert")
            create_bar_chart(dinner_data, 'Type of dish made of rice', "Most Occurring type of dish made of rice", "Count", "Dish made of rice")
            create_bar_chart(dinner_data, 'Salad', "Most Occurring Salad", "Salad", "Dessert")
            create_bar_chart(dinner_data, 'Soup', "Most Occurring soups", "Soups", "Dessert")
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
            plt.xlabel("Dish Combination")
            plt.ylabel("Frequency")
            plt.gca().spines[['top', 'right']].set_visible(False)

            for index, value in enumerate(top_10_dishes):
                plt.text(value + 0.1, index, str(value), va='center', fontweight='bold')
            st.pyplot(plt)
        else:
            st.warning(f"Dataset for {meal_type} is not uploaded.")

    with tab2:
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
    with tab3:
        st.header("Relationship b/w Datasets")

        # Function to extract unique dish suggestions
        def get_dish_suggestions(data, dataset_type):
            if dataset_type == 'Dinner':
                dish_columns = [col for col in data.columns if col.startswith(('Dish', 'Curd', 'Dessert', 'Dal', 'Soup', 'Chapati', 'Chutney', 'Pickle', 'Lemon wedges', 'Chillies', 'Salad', 'Onions', 'Fryums', 'Curry'))]
            elif dataset_type == 'Lunch':
                dish_columns = [col for col in data.columns if col.startswith(('Dish', 'Dal', 'Rice', 'Salad', 'Desert', 'Drinks', 'Chillies', 'Pickle', 'Bread'))]
            elif dataset_type == 'Breakfast':
                dish_columns = [col for col in data.columns if col.startswith('Dish')]
            else:
                return []

            # Collect unique dish names
            unique_dishes = pd.concat([data[col] for col in dish_columns]).dropna().unique()
            return sorted(unique_dishes)

        # Function to analyze dish keyword combinations and return the relevant rows
        def analyze_dish_combinations(dish_keyword, data, dataset_type):
            # Define columns based on dataset type
            if dataset_type == 'Dinner':
                dish_columns = [col for col in data.columns if col.startswith(('Dish', 'Curd', 'Dessert', 'Dal', 'Soup', 'Chapati', 'Chutney', 'Pickle', 'Lemon wedges', 'Chillies', 'Salad', 'Onions', 'Fryums', 'Curry'))]
            elif dataset_type == 'Lunch':
                dish_columns = [col for col in data.columns if col.startswith(('Dish', 'Dal', 'Rice', 'Salad', 'Desert', 'Drinks', 'Chillies', 'Pickle', 'Bread'))]
            elif dataset_type == 'Breakfast':
                dish_columns = [col for col in data.columns if col.startswith('Dish')]
            else:
                return [], {}

            # Filter rows where the dish keyword appears
            relevant_rows = data[data[dish_columns].apply(lambda row: row.str.contains(dish_keyword, case=False, na=False).any(), axis=1)]

            if relevant_rows.empty:
                return [], {}

            # Extract the dates when the dish keyword appears
            dates = relevant_rows['Date']

            return dates, relevant_rows

        # Function to count co-occurrences in the other dataset based on the same dates
        def count_co_occurrences(dish_keyword, dates, other_data, other_dataset_type):
            # Filter rows in the other dataset where the date is in the list of dates from the primary dataset
            filtered_data = other_data[other_data['Date'].isin(dates)]

            # Get the columns that could contain other dishes
            if other_dataset_type == 'Dinner':
                dish_columns = [col for col in other_data.columns if col.startswith(('Dish', 'Curd', 'Dessert', 'Dal', 'Soup', 'Chapati', 'Chutney', 'Pickle', 'Lemon wedges', 'Chillies', 'Salad', 'Onions', 'Fryums', 'Curry'))]
            elif other_dataset_type == 'Lunch':
                dish_columns = [col for col in other_data.columns if col.startswith(('Dish', 'Dal', 'Rice', 'Salad', 'Desert', 'Drinks', 'Chillies', 'Pickle', 'Bread'))]
            else:
                return {}

            # Count occurrences of other dishes on those same dates, excluding the dish keyword
            all_dishes = filtered_data[dish_columns].stack()
            other_dishes_count = all_dishes[~all_dishes.str.contains(dish_keyword, case=False, na=False)].value_counts()

            return other_dishes_count

        # Function to create the network graph
        def network_dish_combinations(dish_keyword, dataset_type, lunch_data, dinner_data):
            # Step 1: Retrieve the dates from the primary dataset (Lunch or Dinner)
            if dataset_type == 'Lunch':
                lunch_dates, _ = analyze_dish_combinations(dish_keyword, lunch_data, 'Lunch')
                dinner_dishes_count = count_co_occurrences(dish_keyword, lunch_dates, dinner_data, 'Dinner')
            elif dataset_type == 'Dinner':
                dinner_dates, _ = analyze_dish_combinations(dish_keyword, dinner_data, 'Dinner')
                lunch_dishes_count = count_co_occurrences(dish_keyword, dinner_dates, lunch_data, 'Lunch')
            else:
                st.error("Invalid dataset selected.")
                return

            # Step 2: Create the graph for co-occurrences based on dates
            G = nx.Graph()

            # Add the input dish as a central node
            central_frequency = len(lunch_dates) if dataset_type == 'Lunch' else len(dinner_dates)
            G.add_node(dish_keyword, size=1000, color='red', label=f"{dish_keyword} (Freq: {central_frequency})")

            # Add other dishes as nodes and create edges for Lunch-Dinner co-occurrences
            if dataset_type == 'Lunch':
                for dish, count in dinner_dishes_count.items():
                    G.add_node(dish, size=count * 100, color='green', label=f"{dish} (Freq: {count})")
                    G.add_edge(dish_keyword, dish, weight=count)
            elif dataset_type == 'Dinner':
                for dish, count in lunch_dishes_count.items():
                    G.add_node(dish, size=count * 100, color='green', label=f"{dish} (Freq: {count})")
                    G.add_edge(dish_keyword, dish, weight=count)

            # Step 3: Generate positions for the nodes
            pos = nx.spring_layout(G, k=0.5, seed=42)

            # Node attributes for the graph
            sizes = [G.nodes[node].get('size', 300) for node in G]
            colors = [G.nodes[node].get('color', 'green') for node in G]
            labels = nx.get_node_attributes(G, 'label')

            # Plot the network graph
            plt.figure(figsize=(10, 6))
            nx.draw(G, pos, with_labels=False, node_size=sizes, node_color=colors, edge_color='gray')

            # Add labels without overlap using `adjust_text`
            texts = []
            for node, (x, y) in pos.items():
                texts.append(plt.text(x, y, labels[node], fontsize=10, ha='center', va='center', color='black'))
            adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))

            # Add a title
            plt.title(f"Network of Dishes Made with '{dish_keyword}'", fontsize=16)
            st.pyplot(plt)

        # Streamlit interactivity
        dataset_type = st.selectbox("Select Dataset Type", ['Lunch', 'Dinner'])
        dish_keyword = st.text_input("Enter Dish Keyword")
        if st.button("Generate Network Graph", key = "button2"):
            if dataset_type == 'Lunch':
                network_dish_combinations(dish_keyword, dataset_type, lunch_data, dinner_data)
            elif dataset_type == 'Dinner':
                network_dish_combinations(dish_keyword, dataset_type, lunch_data, dinner_data)