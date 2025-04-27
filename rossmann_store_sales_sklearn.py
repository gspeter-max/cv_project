
''' 
target ==> ( num_customer, scales)
'''

import polars 

class data_pipeline: 

    def __init__(self,_1st_data : str, _2nd_data : str = '/content/drive/MyDrive/store.csv'):
        self._1st_data = _1st_data
        self._2nd_data = _2nd_data

    def forward(self): 
        
        df1 = polars.read_csv(self._1st_data,ignore_errors= True) 
        df2= polars.read_csv(self._2nd_data, ignore_errors= True)

        df1 = df1.with_columns(
            polars.col('StateHoliday').fill_null(1)
        )

        df1 = df1.with_columns(
            polars.col('Date').str.to_datetime('%Y-%m-%d')
        )
        for i, month_values in enumerate(df2['PromoInterval']):
            month_values = month_values.split(',')
            for sub_month_values in month_values:
                if sub_month_values != '': 
                    if sub_month_values not in df2.columns: 
                        df2 = df2.with_columns(polars.lit(0).alias(sub_month_values))
                    df2[i,sub_month_values] = 1 
        
        df2 = df2.drop('PromoInterval')        
        df2 = df2.with_columns(polars.col('CompetitionDistance').log().alias('CompetitionDistance'))

        def one_hot(df,col): 
            for i,values in enumerate(df[col]): 
                if values not in df.columns: 
                    df = df.with_columns(polars.lit(0).alias(values))

                df[i,values] = 1 
            return df 
        df2 = one_hot(df2,'StoreType')
        df2 = one_hot(df2,'Assortment')
        df2 = df2.drop('StoreType','Assortment')
        final_df= df1.join(
            other = df2, 
            on = 'Store',
            how = 'left'
        )
        return final_df 

train_df = data_pipeline('/content/drive/MyDrive/train_store_data.csv').forward()
