import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans

class Negara:
    AFTA = [
        'BRUNEI DARUSSALAM', 
        'CAMBODIA', 
        'INDONESIA', 
        'LAO, PEOPLE\'S DEMOCRATIC REPUBLIC', 
        'MYANMAR', 
        'PHILIPPINES', 
        'SINGAPORE', 
        'THAILAND', 
        'VIET NAM'
    ]
    
    EUA = [
        'AUSTRIA',
        'BELGIUM',
        'BULGARIA',
        'CROATIA',
        'CYPRUS',
        'CZECH REPUBLIC',
        'DENMARK',
        'ESTONIA',
        'FINLAND',
        'FRANCE',
        'GERMANY',
        'GREECE',
        'HUNGARY',
        'IRELAND',
        'ITALY',
        'LATVIA',
        'LITHUANIA',
        'LUXEMBOURG',
        'MALTA',
        'NETHERLANDS',
        'POLAND',
        'PORTUGAL',
        'ROMANIA',
        'SLOVAKIA',
        'SLOVENIA',
        'SPAIN',
        'SWEDEN'
    ]
    EUA1 = [
        'AUSTRIA',
        'BELGIUM',
        'BULGARIA',
        'CROATIA',
        'CYPRUS',
        'CZECH REPUBLIC',
        'DENMARK'
    ]

    EUA2 = [
        'ESTONIA',
        'FINLAND',
        'FRANCE',
        'GERMANY',
        'GREECE',
        'HUNGARY'
    ]

    EUA3 = [
        'IRELAND',
        'ITALY',
        'LATVIA',
        'LITHUANIA',
        'LUXEMBOURG'
    ]

    EUA4 = [
        'MALTA',
        'NETHERLANDS',
        'POLAND',
        'PORTUGAL',
        'ROMANIA'
    ]

    EUA5 = [
        'SLOVAKIA',
        'SLOVENIA',
        'SPAIN',
        'SWEDEN'
    ]
    
    EFTA = [
        'ICELAND',
        'LIECHTENSTEIN',
        'NORWAY',
        'SWITZERLAND'
    ]
    
    LAIA = [
        'ARGENTINA',
        'BOLIVIA, PLURINATIONAL STATE OF',
        'BRAZIL',
        'CHILE',
        'COLOMBIA',
        'CUBA',
        'ECUADOR',
        'MEXICO',
        'PARAGUAY',
        'PERU',
        'URUGUAY',
        'VENEZUELA, BOLIVARIAN REPUBLIC OF'
    ]
    
    NAFTA = [
        'CANADA',
        'MEXICO',
        'UNITED STATES'
    ]
    
    SAARC = [
        'AFGHANISTAN',
        'BANGLADESH',
        'BHUTAN',
        'INDIA',
        'MALDIVES',
        'NEPAL',
        'PAKISTAN',
        'SRI LANKA'
    ]
    Africa = [
    "ALGERIA", "ANGOLA", "BENIN", "BOTSWANA", "BURKINA FASO", "BURUNDI", "CAMEROON", 
    "CAPE VERDE", "CENTRAL AFRICAN REPUBLIC", "CHAD", "COMOROS", "CONGO", 
    "CONGO, THE DEMOCRATIC REPUBLIC OF THE", "COTE D'IVOIRE", "DJIBOUTI", 
    "EGYPT", "EQUATORIAL GUINEA", "ERITREA", "ESWATINI", "ETHIOPIA", "GABON", 
    "GAMBIA", "GHANA", "GUINEA", "GUINEA-BISSAU", "KENYA", "LESOTHO", "LIBERIA", 
    "LIBYA", "MADAGASCAR", "MALAWI", "MALI", "MAURITANIA", "MAURITIUS", "MAYOTTE", 
    "MOROCCO", "MOZAMBIQUE", "NAMIBIA", "NIGER", "NIGERIA", "REUNION", "RWANDA", 
    "SAINT HELENA, ASCENSION AND TRISTAN DA CUNHA", "SAO TOME AND PRINCIPE", 
    "SENEGAL", "SEYCHELLES", "SIERRA LEONE", "SOMALIA", "SOUTH AFRICA", 
    "SOUTH SUDAN", "SUDAN", "TANZANIA, UNITED REPUBLIC OF", "TOGO", "TUNISIA", 
    "UGANDA", "WESTERN SAHARA", "ZAIRE, REPUBLIC OF", "ZAMBIA", "ZIMBABWE"
    ]
    SouthAmerica = [
        'ANGUILLA',
        'ANTIGUA & BARBUDA',
        'ARGENTINA',
        'ARUBA',
        'BAHAMAS',
        'BARBADOS',
        'BELIZE',
        'BERMUDA',
        'BOLIVIA, PLURINATIONAL STATE OF',
        'BONAIRE, SINT EUSTATIUS AND SABA',
        'BOUVET ISLAND',
        'BRAZIL',
        'CAYMAN ISLANDS',
        'CHILE',
        'COLOMBIA',
        'COSTA RICA',
        'CUBA',
        'CURACAO',
        'DOMINICA',
        'DOMINICAN REPUBLIC',
        'ECUADOR',
        'EL SALVADOR',
        'FALKLAND ISLAND (MALVINAS)',
        'FRENCH GUIANA',
        'GRENADA',
        'GUADELOUPE',
        'GUATEMALA',
        'GUYANA',
        'HAITI',
        'HONDURAS',
        'JAMAICA',
        'MARTINIQUE',
        'MEXICO',
        'MONTSERRAT',
        'NETHERLANDS ANTILLES',
        'NICARAGUA',
        'PANAMA',
        'PARAGUAY',
        'PERU',
        'PUERTO RICO',
        'SAINT KITTS AND NEVIS',
        'SAINT LUCIA',
        'SAINT VINCENT AND THE GRENADINES',
        'SINT MAARTEEN (DUTCH PART)',
        'SURINAME',
        'TRINIDAD AND TOBAGO',
        'UNITED STATES MINOR OUTLYING ISLANDS',
        'URUGUAY',
        'VENEZUELA, BOLIVARIAN REPUBLIC OF',
        'VIRGIN ISLANDS, BRITISH',
        'VIRGIN ISLANDS, U.S'
    ]
    
    NorthAmerica = [
    'CANADA',
    'SAINT BARTHELEMY',
    'SAINT MARTIN (FRENCH PART)',
    'SAINT PIERRE AND MIQUELON',
    'TURKS AND CAICOS ISLANDS',
    'UNITED STATES'
    ]
    
    Asia = [
    'AFGHANISTAN',
    'ARMENIA',
    'AZERBAIJAN',
    'BAHRAIN',
    'BANGLADESH',
    'BHUTAN',
    'BRITISH INDIAN OCEAN TERRITORY',
    'BRUNEI DARUSSALAM',
    'CAMBODIA',
    'CHINA',
    'GEORGIA',
    'HONG KONG',
    'INDIA',
    'INDONESIA',
    'IRAN, ISLAMIC REPUBLIC OF',
    'IRAQ',
    'JAPAN',
    'JORDAN',
    'KAZAKHSTAN',
    'KOREA, REPUBLIC OF',
    'KUWAIT',
    'KYRGYZSTAN',
    'LAO, PEOPLE\'S DEMOCRATIC REPUBLIC',
    'LEBANON',
    'MACAO',
    'MALDIVES',
    'MONGOLIA',
    'MYANMAR',
    'NEPAL',
    'NEUTRAL ZONE',
    'OMAN',
    'PAKISTAN',
    'PALESTINIAN TERRITORY OCCUPIED',
    'PHILIPPINES',
    'QATAR',
    'SAUDI ARABIA',
    'SINGAPORE',
    'SOUTH GEORGIA AND THE SOUTH SANDWICH ISLANDS',
    'SRI LANKA',
    'SYRIAN ARAB REPUBLIC',
    'TAIWAN, PROVINCE OF CHINA',
    'TAJIKISTAN',
    'THAILAND',
    'TIMOR LESTE',
    'TURKMENISTAN',
    'UNITED ARAB EMIRATES',
    'UZBEKISTAN',
    'VIET NAM',
    'YEMEN'
    ]
    
    Europe = [
    'ALAND ISLANDS',
    'ALBANIA',
    'ANDORRA',
    'AUSTRIA',
    'BELARUS',
    'BELGIUM',
    'BOSNIA AND HERZEGOVINA',
    'BULGARIA',
    'CROATIA',
    'CYPRUS',
    'CZECH REPUBLIC',
    'DENMARK',
    'ESTONIA',
    'FAROE ISLANDS',
    'FINLAND',
    'FRANCE',
    'GERMANY',
    'GIBRALTAR',
    'GREECE',
    'GREENLAND',
    'GUERNSEY',
    'HOLY SEE (VATICAN CITY STATE)',
    'HUNGARY',
    'ICELAND',
    'IRELAND',
    'ITALY',
    'JERSEY',
    'LATVIA',
    'LIECHTENSTEIN',
    'LITHUANIA',
    'LUXEMBOURG',
    'MACEDONIA, THE FORMER YUGOSLAV REPUBLIC OF',
    'MALTA',
    'MOLDOVA, REPUBLIC OF',
    'MONACO',
    'MONTENEGRO',
    'NETHERLANDS',
    'NORWAY',
    'POLAND',
    'PORTUGAL',
    'ROMANIA',
    'RUSSIAN FEDERATION',
    'SAN MARINO',
    'SERBIA',
    'SLOVAKIA',
    'SLOVENIA',
    'SPAIN',
    'SVALBARD AND JAN MAYEN',
    'SWEDEN',
    'SWITZERLAND',
    'TURKIYE',
    'UKRAINE',
    'UNITED KINGDOM',
    'YUGOSLAVIA, FED REP OF'
    ]
    
    Oceania = [
    'AMERICAN SAMOA',
    'AUSTRALIA',
    'CHRISTMAS ISLAND',
    'COCOS (KEELING) ISLANDS',
    'COOK ISLANDS',
    'FIJI',
    'FRENCH POLYNESIA',
    'FRENCH SOUTHERN TERRITORIES',
    'GUAM',
    'HEARD ISLAND AND MCDONALD ISLANDS',
    'KIRIBATI',
    'MARSHALL ISLANDS',
    'MICRONESIA, FEDERATED STATES OF',
    'NAURU',
    'NEW CALEDONIA',
    'NEW ZEALAND',
    'NIUE',
    'NORFOLK ISLAND',
    'NORTHERN MARIANA ISLANDS',
    'PALAU',
    'PAPUA NEW GUINEA',
    'PITCAIRN',
    'SOLOMON ISLANDS',
    'TOKELAU',
    'TONGA',
    'TUVALU',
    'VANUATU',
    'WALLIS AND FUTUNA',
    'WESTERN SAMOA'
    ]





    
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.exports_grouped = None
        self.imports_grouped = None
        self.grp = None
    
    def load_and_clean_data(self):
        try:
            # Load the data
            self.df = pd.read_csv(self.file_path)
            
            # Clean the data
            self.df.replace(r'\s+', '', regex=True, inplace=True)  # Remove extra spaces
            self.df.replace({',': '', '"': ''}, regex=True, inplace=True)  # Remove commas and quotes
            self.df.replace({'*': np.nan, '-': np.nan}, inplace=True)  # Replace '*' and '-' with NaN
            
            # Keep rows where 'NEGARA' column is not missing
            self.df = self.df[self.df['NEGARA'].notna()]
            
            # Reset index if needed
            self.df.reset_index(drop=True, inplace=True)
        except FileNotFoundError:
            print(f"Error: The file at {self.file_path} was not found.")
        except pd.errors.EmptyDataError:
            print("Error: The file is empty.")
        except pd.errors.ParserError:
            print("Error: There was an issue parsing the file.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    
    def extract_and_group_data(self):
        try:
            # Extracting the exports and imports data
            exports = self.df[['NEGARA', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']]
            imports = self.df[['COUNTRY', '2016.1', '2017.1', '2018.1', '2019.1', '2020.1', '2021.1', '2022.1', '2023.1']]
            
            # Renaming columns for clarity
            exports.columns = ['Exports_Country', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
            imports.columns = ['Imports_Country', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
            
            # Grouping the data by country and summing up the values
            self.exports_grouped = exports.groupby('Exports_Country').sum()
            self.imports_grouped = imports.groupby('Imports_Country').sum()
        except KeyError as e:
            print(f"Error: Missing expected column in the data - {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
    def sum_exports_imports(self):
        # Selecting relevant columns for exports and imports
        exports = self.df[['NEGARA', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']]
        imports = self.df[['COUNTRY', '2016.1', '2017.1', '2018.1', '2019.1', '2020.1', '2021.1', '2022.1', '2023.1']]

        # Renaming columns for clarity
        exports.columns = ['Negara', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
        imports.columns = ['Negara', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']

        exports[['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']] = exports[['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']].astype(float)
        imports[['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']] = imports[['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']].astype(float)

        # Calculating the sum of exports and imports for the years 2016 to 2023
        exports['Total_Exports'] = exports[['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']].sum(axis=1)
        imports['Total_Imports'] = imports[['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']].sum(axis=1)

        # Merging exports and imports into a new DataFrame
        self.grp = pd.merge(exports[['Negara', 'Total_Exports']], 
                            imports[['Negara', 'Total_Imports']], 
                            on='Negara', 
                            how='outer')
        self.grp.set_index('Negara', inplace=True)

        return self.grp  

    
    def get_exports_grouped(self):
        return self.exports_grouped
    
    def get_imports_grouped(self):
        return self.imports_grouped
    
    def get_grp(self):
        return self.grp
    
    def createImportGrouped(self, grp_list):
        try:
            # Filter the imports_grouped DataFrame based on countries in the grp_list
            grp_import = self.imports_grouped.loc[self.imports_grouped.index.isin(grp_list)]
            return grp_import
        except KeyError as e:
            print(f"Error: Key not found in imports_grouped - {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    
    def createExportGrouped(self, grp_list):
        try:
            # Filter the exports_grouped DataFrame based on countries in the grp_list
            grp_export = self.exports_grouped.loc[self.exports_grouped.index.isin(grp_list)]
            return grp_export
        except KeyError as e:
            print(f"Error: Key not found in exports_grouped - {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
    def createGrouped(self, countries_list):
        # Ensure that self.grp has been calculated
        if self.grp is None:
            raise ValueError("The sum_exports_imports method must be called before createGrouped.")

        # Filtering the DataFrame based on the provided countries list
        grouped_df = self.grp[self.grp.index.isin(countries_list)]

        return grouped_df


    def calNet(self, exports_grouped, imports_grouped):
        try:
            # Rename imports_grouped index to match exports_grouped index
            imports_grouped.index.name = 'Exports_Country'

            # Merge the two DataFrames on the index (country names)
            combined_df = pd.merge(exports_grouped, imports_grouped, left_index=True, right_index=True, suffixes=('_Exports', '_Imports'))

            # Convert all columns to float
            combined_df = combined_df.astype(float)

            # Calculate net exports for each year
            for year in exports_grouped.columns:
                combined_df[year + '_Net'] = combined_df[year + '_Exports'] - combined_df[year + '_Imports']

            # Drop columns with '_Exports' and '_Imports' suffixes
            columns_to_drop = combined_df.filter(like='_Exports').columns.tolist() + combined_df.filter(like='_Imports').columns.tolist()
            combined_df = combined_df.drop(columns=columns_to_drop)

            return combined_df 
        except KeyError as e:
            print(f"Error: Key not found in combined DataFrame - {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    
    def plot_elbow_method(self, result_df):
        try:
            # Get the number of samples
            n_samples = result_df.shape[0]

            # Define a reasonable upper limit for the number of clusters
            max_clusters = min(n_samples, 10)  # For example, 10 or less if n_samples is less than 10

            clustercol = []

            for i in range(1, max_clusters + 1):
                km = KMeans(n_clusters=i, random_state=0)  # Set random_state for reproducibility
                km.fit_predict(result_df)  
                clustercol.append(km.inertia_)  

            # Create the plot
            fig, ax = plt.subplots()
            ax.plot(range(1, max_clusters + 1), clustercol, '-o')  # Marker = '-o' for circle
            ax.set_xlabel('Number of Clusters')
            ax.set_ylabel('Inertia')
            ax.set_title('Elbow Method for Optimal Number of Clusters')
            plt.show()
        except ValueError as e:
            print(f"Error: Value error in KMeans - {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    
    def perform_clustering(self, result_df, n_clusters):
        try:
            # Initialize KMeans with the specified number of clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=40)  # random_state for reproducibility
            kmeans.fit(result_df)  # Fit the model to the data

            # Predict the cluster labels
            result_df['clusters'] = kmeans.predict(result_df)

            # Calculate the mean of each cluster
            cluster_means = result_df.groupby('clusters').mean()

            return cluster_means
        except ValueError as e:
            print(f"Error: Value error in KMeans - {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
    def plot_net_exports(self, df, title, savefig_name):
        # Ensure the DataFrame is sorted by index (e.g., country names)
        df = df.sort_index()

        # Create a new figure for plotting
        plt.figure(figsize=(12, 8))  # Adjust the size as needed

        # Plot each year as a set of horizontal bars
        for column in df.columns:
            plt.barh(df.index, df[column], label=column)

        # Set the labels and title
        plt.xlabel('Net Exports')
        plt.ylabel('Country')
        plt.title(title)
        plt.legend(title='Year', loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()  # Adjust layout to fit labels

        # Save the plot as an image file
        plt.savefig(savefig_name)  # Save the plot as an image
        plt.close()  # Close the plot to free up resources
        
    def plot_net_exports_line(self, df, title, savefig_name):
        # Ensure the DataFrame is sorted by index (e.g., country names)
        df = df.sort_index()

        # Create a color palette with at least 26 colors
        # Seaborn's color_palette() can be used with the 'husl' or 'tab20' palette, or you can create your own
        colors = sns.color_palette("tab20", n_colors=len(df.index))

        # Create a new figure for plotting
        plt.figure(figsize=(12, 6))  # Adjust the size as needed

        # Plot each country as a line with a specific color
        for i, country in enumerate(df.index):
            plt.plot(df.columns, df.loc[country], marker='o', label=country, color=colors[i])

        # Set the labels and title
        plt.xlabel('Year')
        plt.ylabel('Net Exports')
        plt.title(title)

        # Set legend to be outside the plot
        plt.legend(title='Country', loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside to the right

        plt.xticks(rotation=45)  # Rotate year labels for better readability
        plt.tight_layout()  # Adjust layout to fit labels

        # Save the plot as an image file
        plt.savefig(savefig_name, bbox_inches='tight')  # Save the plot with the adjusted layout
        plt.close()  # Close the plot to free up resources
        
    def find_max_min(self, df):
        try:
            # Ensure DataFrame is not empty
            if df.empty:
                raise ValueError("The DataFrame is empty")

            # Initialize dictionaries to hold the results
            max_values = {}
            min_values = {}
            max_countries = {}
            min_countries = {}
            max_years = {}
            min_years = {}

            # Iterate over each year (column) in the DataFrame
            for year in df.columns:
                # Find maximum and minimum values for the year
                max_value = df[year].max()
                min_value = df[year].min()

                # Find the country corresponding to these values
                max_country = df[df[year] == max_value].index.tolist()
                min_country = df[df[year] == min_value].index.tolist()

                # Store the results in the dictionaries
                max_values[year] = max_value
                min_values[year] = min_value
                max_countries[year] = max_country
                min_countries[year] = min_country
                

            # Create DataFrames for max and min values
            max_df = pd.DataFrame({
                'Max Value': max_values,
                'Max Country': max_countries,
                
            })

            min_df = pd.DataFrame({
                'Min Value': min_values,
                'Min Country': min_countries,
                
            })

            # Return the results as a dictionary containing DataFrames
            return {
                'Max Values': max_df,
                'Min Values': min_df
            }
        except ValueError as e:
            print(f"Error: {e}")
        except KeyError as e:
            print(f"Error: Key not found in DataFrame - {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def find_mx(self,df):
        try:
            # Ensure DataFrame is not empty
            if df.empty:
                raise ValueError("The DataFrame is empty")

            # Initialize dictionaries to hold the results
            max_values = {}
            max_countries = {}

            # Iterate over each year (column) in the DataFrame
            for year in df.columns:
                # Find maximum values for the year
                max_value = df[year].max()

                # Find the countries corresponding to these values
                max_countries[year] = df[df[year] == max_value].index.tolist()

                # Store the results in the dictionaries
                max_values[year] = max_value

            # Create DataFrames for max and min values
            max_df = pd.DataFrame({
                'Year': list(max_values.keys()),
                'Max Value': list(max_values.values()),
                'Max Country': [', '.join(countries) for countries in max_countries.values()]
            }).set_index('Year')

            # Return the results as a dictionary containing DataFrames
            return max_df
            
        except ValueError as e:
            print(f"Error: {e}")
        except KeyError as e:
            print(f"Error: Key not found in DataFrame - {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def find_mn(self,df):
        try:
            # Ensure DataFrame is not empty
            if df.empty:
                raise ValueError("The DataFrame is empty")

            # Initialize dictionaries to hold the results
            min_values = {}
            min_countries = {}

            # Iterate over each year (column) in the DataFrame
            for year in df.columns:
                min_value = df[year].min()
                min_countries[year] = df[df[year] == min_value].index.tolist()
                min_values[year] = min_value

            # Create a DataFrame from the minimum values and countries
            min_df = pd.DataFrame({
                'Year': list(min_values.keys()),
                'Min Value': list(min_values.values()),
                'Min Country': [', '.join(countries) for countries in min_countries.values()]
            }).set_index('Year')

            # Return the DataFrame
            return min_df

        except ValueError as e:
            print(f"Error: {e}")
            return None
        except KeyError as e:
            print(f"Error: Key not found in DataFrame - {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
    @staticmethod
    def plot_net_exports_eua(self,df, title, savefig_name):
        # Ensure the DataFrame is sorted by index (e.g., country names)
        df = df.sort_index()

        # Create a color palette with at least 26 colors
        colors = sns.color_palette("tab20", n_colors=len(df.index))

        # Create a new figure for plotting
        plt.figure(figsize=(12, 6))  # Adjust the size as needed

        # Plot each country as a line with a specific color
        for i, country in enumerate(df.index):
            plt.plot(df.columns, df.loc[country], marker='o', label=country, color=colors[i])

        # Set the labels and title
        plt.xlabel('Year')
        plt.ylabel('Net Exports')
        plt.title(title)

        # Set legend to be outside the plot
        plt.legend(title='Country', loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside to the right

        plt.xticks(rotation=45)  # Rotate year labels for better readability
        plt.tight_layout()  # Adjust layout to fit labels

        # Save the plot as an image file
        plt.savefig(savefig_name, bbox_inches='tight')  # Save the plot with the adjusted layout
        plt.close()  # Close the plot to free up resources
        
 