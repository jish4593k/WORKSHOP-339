# -*- coding: utf-8 -*-
"""
author SparkByExamples.com
"""

import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import col
from tkinter import *
from tkinter import ttk

# Create a Pandas DataFrame
data = [['Scott', 50], ['Jeff', 45], ['Thomas', 54], ['Ann', 34]]
pandasDF = pd.DataFrame(data, columns=['Name', 'Age'])

# Create a PySpark DataFrame
spark = SparkSession.builder.master("local[1]").appName("SparkByExamples.com").getOrCreate()
mySchema = StructType([StructField("First Name", StringType(), True), StructField("Age", IntegerType(), True)])
sparkDF = spark.createDataFrame(pandasDF, schema=mySchema)

# Print PySpark DataFrame Schema and Show Data
print("PySpark DataFrame Schema:")
sparkDF.printSchema()
print("PySpark DataFrame:")
sparkDF.show()

# Convert PySpark DataFrame to Pandas DataFrame
pandasDF2 = sparkDF.select("*").toPandas()

# Visualize Data using Seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(8, 4))
sns.barplot(x="Age", y="First Name", data=pandasDF2)
plt.title("Age Distribution")
plt.show()

# Use Torch to create a Tensor
age_tensor = torch.tensor(pandasDF2['Age'].values, dtype=torch.float32)

# GUI: Display Data in a Tkinter Window
def show_data():
    top = Tk()
    top.title("PySpark DataFrame in Tkinter")

    tree = ttk.Treeview(top)
    tree["columns"] = ("Name", "Age")
    tree.column("#0", width=0, stretch=NO)
    tree.column("Name", anchor=W, width=100)
    tree.column("Age", anchor=W, width=100)

    tree.heading("#0", text="", anchor=W)
    tree.heading("Name", text="Name", anchor=W)
    tree.heading("Age", text="Age", anchor=W)

    for index, row in pandasDF2.iterrows():
        tree.insert("", index, values=(row['First Name'], row['Age']))

    tree.pack()
    top.mainloop()


# Run the Tkinter GUI
show_data()

# Spark Configuration
arrow_enabled = spark.conf.get("spark.sql.execution.arrow.enabled")
fallback_enabled = spark.conf.get("spark.sql.execution.arrow.pyspark.fallback.enabled")

# Use different logic based on Spark configurations
if arrow_enabled == "true":
    # Perform some logic when Arrow is enabled
    print("Arrow is enabled")
else:
    # Perform some alternative logic
    print("Arrow is not enabled")

if fallback_enabled == "true":
    # Perform some logic when Arrow fallback is enabled
    print("Arrow fallback is enabled")
else:
    # Perform some alternative logic
    print("Arrow fallback is not enabled")
