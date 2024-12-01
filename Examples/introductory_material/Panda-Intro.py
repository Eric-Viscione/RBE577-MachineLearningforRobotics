import pandas as pd
import numpy as np

s = pd.Series([1, 3, 5, np.nan, 6, 8])
#print(s)
dates = pd.date_range("20130101", periods=6)
#print(dates)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
#print(df)
df2 = pd.DataFrame(

    {

        "A": 1.0,

        "B": pd.Timestamp("20130102"),

        "C": pd.Series(1, index=list(range(4)), dtype="float32"),

        "D": np.array([3] * 4, dtype="int32"),

        "E": pd.Categorical(["test", "train", "test", "train"]),

        "F": "foo",

    }

)
#print(df2)
#print("////")
#print(df2.dtypes)
#print(df.head(1))
#print(df.tail(3))
#print(df.index)
#print(df.columns)
#print(df.to_numpy())
#print(df2.to_numpy())
#print(df.describe)
#print(df.T)
#print(df.sort_index(axis=1, ascending=False))
#print(df.sort_values(by="B"))
# print(df["A"])
# print(df[0:3])
# print(df["20130102":"20130104"])
# print(df.loc[dates[0]])
# print(df.loc[:, ["A", "B"]])
# print(df.loc["20130102":"20130104", ["A", "B"]])
# print(df.loc[dates[0], "A"])
# print(df.at[dates[0], "A"])
# print(df.iloc[3])
# print(df[df["A"] > 0])
# df2 = df.copy()
# df2["E"] = ["one", "one", "two", "three", "four", "three"]  
# print(df2)
# print(df2[df2["E"].isin(["two", "four"])])
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range("20130102", periods=6))
df["F"] = s1
# print(df)

df.at[dates[0], "A"] = 0
# print(df)
df.iat[0, 1] = 0
# print(df)
df.loc[:, "D"] = np.array([5] * len(df))
# print(df)
df2 = df.copy()
# print(df2)
df2[df2 > 0] = -df2
# print(df2)
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ["E"])
# print(df.mean())
# print(df.mean(axis=1))
s = pd.Series(np.random.randint(0, 7, size=10))
# print(s.value_counts())
s = pd.Series(["A", "B", "C", "Aaba", "Baca", np.nan, "CABA", "dog", "cat"])
s.str.lower()
df = pd.DataFrame(np.random.randn(10, 4))
pieces = [df[:3], df[3:7], df[7:]]

df = pd.concat(pieces)
# print(df)

