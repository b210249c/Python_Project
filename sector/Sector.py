import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class SectorAnalysis:
    def __init__(self):
        self.df = None

    # Function to load the data from a CSV file
    def load_data(self, filepath):
        try:
            df = pd.read_csv(filepath)
            print(f"Successfully loaded the file: {filepath}")
            return df
        except FileNotFoundError as fnf_error:
            print(f"Error: {fnf_error}. The file was not found. Please check the file path and name.")
        except pd.errors.EmptyDataError:
            print("Error: The file is empty. Please check the content of the file.")
        except pd.errors.ParserError:
            print("Error: There was an issue with parsing the data. Please check the file format.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        return None

    
    # Function to load and clean the sector data, handling specific rows
    def load_and_clean_sector_data(self, filepath):
        try:
            # Load the CSV file, skipping the first row
            df = pd.read_csv(filepath, skiprows=1)
            
            # Drop the first row (index 0)
            df = df.drop(0)
            
            # Reset index without keeping the old one
            df.reset_index(drop=True, inplace=True)
            
            print(f"Successfully loaded and cleaned the file: {filepath}")
            return df
    
        except FileNotFoundError as fnf_error:
            print(f"Error: {fnf_error}. The file was not found. Please check the file path and name.")
        except pd.errors.EmptyDataError:
            print("Error: The file is empty. Please check the content of the file.")
        except pd.errors.ParserError:
            print("Error: There was an issue with parsing the data. Please check the file format.")
        except KeyError as ke:
            print(f"Error: {ke}. One of the columns to rename was not found. Check the column names.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        
        return None

    
    # Function to split the data based on sectors 
    def split_sectors(self, df, sector):
        try:
            # Determine the starting indices for different sectors
            agriculture_start_index = df[df['SECTOR / SELETED COMMODITIES'] == 'AGRICULTURE'].index[0]
            mining_start_index = df[df['SECTOR / SELETED COMMODITIES'] == 'MINING'].index[0]
            manufacturing_start_index = df[df['SECTOR / SELETED COMMODITIES'] == 'MANUFACTURING'].index[0]
    
            # Slice the data based on the specified sector and reset the index
            if sector == 'AGRICULTURE':   
                new_df = df.iloc[agriculture_start_index:mining_start_index]
            elif sector == 'MINING': 
                new_df = df.iloc[mining_start_index:manufacturing_start_index]
            else:
                new_df = df.iloc[manufacturing_start_index:]

            new_df.reset_index(drop=True, inplace=True)
            
            return new_df
    
        except FileNotFoundError:
            print(f"Error: The file '{csv_file}' was not found.")
        except pd.errors.EmptyDataError:
            print(f"Error: The file '{csv_file}' is empty.")
        except pd.errors.ParserError:
            print(f"Error: There was an error parsing the file '{csv_file}'.")
        except KeyError as e:
            print(f"Error: The column {e} was not found in the CSV file.")
        except IndexError:
            print(f"Error: One of the sectors 'AGRICULTURE', 'MINING', or 'MANUFACTURING' was not found.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    
    # Function to clean the sector-specific DataFrame, renaming columns and filtering rows
    def clean_sector_df(self, df):
        try:
            # Rename the column
            df = df.rename(columns={'SECTOR / SELETED COMMODITIES': 'SELETED COMMODITIES'})
            df = df.rename(columns={'2021.0': 'exp_2021', '2022.0': 'exp_2022', '2023.0': 'exp_2023',
                                    '2021.1': 'imp_2021', '2022.1': 'imp_2022', '2023.1': 'imp_2023'})
            # remove row
            df = df[df['SELETED COMMODITIES'] != 'OTHERS']
            df = df[df['SELETED COMMODITIES'] != df['SELETED COMMODITIES'].iloc[0]]
            
            return df
        except Exception as e:
            print(f"Error cleaning sector data: {e}")
            return None

    
    # Function to add a sector column to the DataFrame and calculate net trade values
    def add_sector(self, name, df):
        try:
            df = self.clean_sector_df(df)

            # Convert DataFrame rows to a list of dictionaries and add the new column
            list_of_dicts = []
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                row_dict['net_2021'] = row_dict['exp_2021'] - row_dict['imp_2021']
                row_dict['net_2022'] = row_dict['exp_2022'] - row_dict['imp_2022']
                row_dict['net_2023'] = row_dict['exp_2023'] - row_dict['imp_2023']
                row_dict['SECTOR'] = name
                list_of_dicts.append(row_dict)

            # Create a new DataFrame from the list of dictionaries
            new_df = pd.DataFrame(list_of_dicts)
            return new_df
        except Exception as e:
            print(f"Error adding sector data: {e}")
            return None

    
    # Function to calculate the cumulative growth rate for exports and imports
    def calculate_cumulative_growth_rate(self, df):
        try:
            exp_growth_2021_2022 = df["exp_2022"] / df["exp_2021"] - 1
            exp_growth_2022_2023 = df["exp_2023"] / df["exp_2022"] - 1
            imp_growth_2021_2022 = df["imp_2022"] / df["imp_2021"] - 1
            imp_growth_2022_2023 = df["imp_2023"] / df["imp_2022"] - 1

            df["Cumulative_Export_Growth_Rate_2021_2023"] = ((1 + exp_growth_2021_2022) * (1 + exp_growth_2022_2023) - 1) * 100
            df["Cumulative_Import_Growth_Rate_2021_2023"] = ((1 + imp_growth_2021_2022) * (1 + imp_growth_2022_2023) - 1) * 100
            return df
        except Exception as e:
            print(f"Error calculating cumulative growth rate: {e}")
            return None

    
    # Function to create a DataFrame containing cumulative growth rates for easier visualization
    def create_df_cumulative_growth_rate(self, df):
        try:
            growth_df = self.calculate_cumulative_growth_rate(df)
            return growth_df[['SELETED COMMODITIES','Cumulative_Export_Growth_Rate_2021_2023','Cumulative_Import_Growth_Rate_2021_2023']]
        except Exception as e:
            print(f"Error creating cumulative growth rate DataFrame: {e}")
            return None


    # Function to plot a heatmap of the cumulative growth rates
    def plot_cumulative_growth_rate(self, df):
        df.set_index('SELETED COMMODITIES', inplace=True)

        plt.figure(figsize=(11,13))
        sns.heatmap(df, annot=True, cmap='coolwarm', fmt=".1f", linewidths=.5)
        plt.title('Cumulative Growth Rates (%) of All Selected Commodities (2021-2023)')
        plt.ylabel('Selected Commodities')
        return plt.show()


    # Function to create a DataFrame containing net trade values for multiple years
    def create_df_net(self, df):
        try:
            return df[['SELETED COMMODITIES','net_2021','net_2022','net_2023']]
        except Exception as e:
            print(f"Error creating net DataFrame: {e}")
            return None


    # Function to plot trade balance trends over multiple years for a specific sector
    def plot_trade_balance(self, df, name):
        try:
            years = ['net_2021', 'net_2022', 'net_2023']
            colors = ['RoyalBlue', 'MediumSeaGreen', 'LightSalmon']
            for i, year in enumerate(years):
                plt.figure(figsize=(11, 6))
                plt.plot(df['SELETED COMMODITIES'], df[year], color=colors[i], marker='o', linestyle='-', label=year)
                for x, y in zip(df['SELETED COMMODITIES'], df[year]):
                    plt.text(x, y, round(y, 2), ha='center', va='bottom', fontsize=7, color='black', fontweight='bold')
                plt.title(f'Trade Balance Trends of Selected {name} Commodities ({year[-4:]})')
                plt.ylabel('Trade Balance (Millions)')
                plt.xlabel('Selected Commodities')
                plt.xticks(rotation=-90)
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"Error plotting trade balance: {e}")


    # Function to find the highest and lowest export/import values across multiple years
    def find_highest_lowest(self, df, column_prefix):
        try:
            years = ['2021', '2022', '2023']
            result = {}
            for year in years:
                col_name = f"{column_prefix}_{year}"
                if col_name in df.columns:
                    highest = df.loc[df[col_name].idxmax()]
                    lowest = df.loc[df[col_name].idxmin()]
                    result[year] = {
                        'highest': {'commodity': highest['SELETED COMMODITIES'], 'value': highest[col_name]},
                        'lowest': {'commodity': lowest['SELETED COMMODITIES'], 'value': lowest[col_name]}
                    }
                else:
                    result[year] = {
                        'highest': {'commodity': None, 'value': None},
                        'lowest': {'commodity': None, 'value': None}
                    }
            return result
        except Exception as e:
            print(f"Error finding highest and lowest: {e}")
            return None


    # Function to plot the category (highest/lowest) data
    def plot_category(self, data, category, title, color):
        try:
            years = ['2021', '2022', '2023']
            commodities = [data[year][category]['commodity'] for year in years]
            values = [data[year][category]['value'] for year in years]

            plt.figure(figsize=(12, 6)) 
            bars = plt.bar(years, values, color=color)
            plt.title(title)
            plt.xlabel('Year')
            plt.ylabel('Value (Millions)')

            # Display the commodity name and value centered on the bars
            for i, (commodity, value) in enumerate(zip(commodities, values)):
                if commodity is not None and value is not None:
                    plt.text(i, value / 2, f"{commodity}\n ({value:.2f})", ha='center', va='center', color='black', fontweight='bold')
                else:
                    plt.text(i, 0, "No Data", ha='center', va='center', color='black', fontweight='bold')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting category: {e}")

    
    # Analyzes the highest and lowest export and import commodities for a given sector, then plots the results in separate categories.
    def analyze_and_plot_highest_lowest(self, df, name):
        try:
            # Analyzing the data
            export_result = self.find_highest_lowest(df, 'exp')
            import_result = self.find_highest_lowest(df, 'imp')
    
            # Plotting the results
            self.plot_category(export_result, 'highest', f'Highest Export Commodities Comparing {name} Sectors (Each Year)', 'skyblue')
            self.plot_category(export_result, 'lowest', f'Lowest Export Commodities Comparing {name} Sectors (Each Year)', 'lightcoral')
            self.plot_category(import_result, 'highest', f'Highest Import Commodities Comparing {name} Sectors (Each Year)', 'lightgreen')
            self.plot_category(import_result, 'lowest', f'Lowest Import Commodities Comparing {name} Sectors (Each Year)', 'orange')
    
        except KeyError as ke:
            print(f"KeyError: The DataFrame for {name} is missing expected keys: {ke}")
        except ValueError as ve:
            print(f"ValueError: Invalid value encountered in {name}: {ve}")
        except TypeError as te:
            print(f"TypeError: Type mismatch in {name}: {te}")
        except Exception as e:
            print(f"An unexpected error occurred in {name}: {e}")


    # Identifies the most and least profitable commodities for each year in the given DataFrame
    def find_profit_loss(self, df, column_prefix):
        try:
            result = {}
            years = ['2021', '2022', '2023']
            for year in years:
                col_name = f"{column_prefix}_{year}"
                most_profitable = df.loc[df[col_name].idxmax()]
                least_profitable = df.loc[df[col_name].idxmin()]
                result[year] = {
                    'most_profitable': {'commodity': most_profitable['SELETED COMMODITIES'], 'value': most_profitable[col_name]},
                    'least_profitable': {'commodity': least_profitable['SELETED COMMODITIES'], 'value': least_profitable[col_name]}
                }
            return result
        except Exception as e:
            print(f"Error finding profit and loss: {e}")
            return None


    # Plots a bar chart showing the most and least profitable commodities over multiple years
    def plot_profit_loss_graph(self, result, name):
        try:
            years = ['2021', '2022', '2023']  # Define the years for the x-axis
            most_profitable_values = [result[year]['most_profitable']['value'] for year in years]
            least_profitable_values = [result[year]['least_profitable']['value'] for year in years]
            most_profitable_labels = [result[year]['most_profitable']['commodity'] for year in years]
            least_profitable_labels = [result[year]['least_profitable']['commodity'] for year in years]
    
            fig, ax = plt.subplots(figsize=(10, 6))

            # Most profitable projects
            ax.bar(years, most_profitable_values, color='green', label='Most Profitable')
            # Least profitable projects
            ax.bar(years, least_profitable_values, color='red', label='Least Profitable')

            # Adding labels for the bars
            for i, year in enumerate(years):
                # Annotation for most profitable commodity
                most_label = f"{most_profitable_labels[i]} ({most_profitable_values[i]:,.2f})"
                ax.text(i, most_profitable_values[i] + 1000, most_label, ha='center', color='black', fontweight='bold', fontsize=7)
                
                # Annotation for least profitable commodity
                least_label = f"{least_profitable_labels[i]} ({least_profitable_values[i]:,.2f})"
                ax.text(i, least_profitable_values[i] -3000, least_label, ha='center', color='black', fontweight='bold', fontsize=7)
    
            ax.set_ylabel('Net Export Value (Millions)')
            ax.set_title(f'Most and Least Profitable {name} Commodities (2021-2023)')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.show()
        except Exception as e:
            print(f"Error plotting graph: {e}")


    # Plots a scatter plot of clustered sectors based on export growth, import growth, and trade balance
    def plot_clusters(self, df):
        try:
            # Prepare data for clustering
            features = df[['exp_growth', 'imp_growth', 'trade_balance']]
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Apply KMeans Clustering
            kmeans = KMeans(n_clusters=4, random_state=42)
            df['cluster'] = kmeans.fit_predict(scaled_features)
            
            # Plotting clusters
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='exp_growth', y='imp_growth', hue='cluster', data=df, palette='Set1')
            plt.title('Clustering of Sectors Based on Export/Import Growth and Trade Balance')
            plt.xlabel('Export Growth (%)')
            plt.ylabel('Import Growth (%)')
            plt.grid(True)
            plt.show()

        except KeyError as e:
            print(f"Error: Missing column in DataFrame: {e}")
        except ValueError as e:
            print(f"Error: Value issue encountered: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


    # write data frame to csv file
    def write_csv_file(self, filepath, df):
        try:
            df.to_csv(filepath, index=False)
            print(f"DataFrame written to {filepath} successfully.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")