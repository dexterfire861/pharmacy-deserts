import pandas as pd 



def read_financial_data(file_path):
    df = pd.read_csv(file_path)
    #print(df.columns)
    df = df[['NAME', 'S1901_C01_012E']]
    return df

def read_population_data(file_path):
    df = pd.read_csv(file_path)

    #print(df.columns)
    df = df.iloc[:, [1, 2, 3]]
    return df

def read_pharmacy_data(file_path):
    df = pd.read_csv(file_path)
    #print(df.columns)
    df = df[['ZIP', 'NAME', 'X', 'Y']]
    return df

def read_health_data(file_path):
    df = pd.read_csv(file_path)
    #print(df.columns)
    df = df[['ZCTA5', 'GHLTH_CrudePrev']]
    return df



def main():
    financial_data = read_financial_data('data/financial_data.csv')
    health_data = read_health_data('data/health_data.csv')
    pharmacy_data = read_pharmacy_data('data/pharmacy_data.csv')
    population_data = read_population_data('data/population_data.csv')

    

    

if __name__ == "__main__":
    main()