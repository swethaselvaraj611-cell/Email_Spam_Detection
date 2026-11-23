import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(
    "SMSSpamCollection.txt",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print(data.head())

print("Total messages:", len(data))

print("\nLabel counts:")
print(data["label"].value_counts())

print("\nMissing values:")
print(data.isnull().sum())

counts = data["label"].value_counts()

counts.plot(kind="bar")

plt.title("Spam vs Ham Messages")
plt.xlabel("Message Type")
plt.ylabel("Number of Messages")

plt.show()