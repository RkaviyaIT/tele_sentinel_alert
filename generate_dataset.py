import pandas as pd
import random

# List of suspicious and non-suspicious messages (you can expand these lists)
suspicious_messages = [
    "Get rich quick with this investment opportunity!",
    "Free crypto giveaway, send your wallet address!",
    "Earn money without doing anything, sign up now!",
    "Investment in crypto is guaranteed to make you rich!",
    "Claim your prize by sending your details now!",
    "Hurry up! Limited offer to make quick money!",
    "Click here to unlock your free vacation prize!",
    "Pay now and double your money instantly!",
    "Invest in this high-return crypto project!",
    "Exclusive offer to join the crypto revolution!"
]

non_suspicious_messages = [
    "Hello, how are you?",
    "Let's grab coffee sometime!",
    "Good morning, everyone!",
    "Did you see the new movie last night?",
    "Can you recommend a good restaurant?",
    "Looking forward to our meeting tomorrow.",
    "How was your weekend?",
    "I'm going to the gym later, want to join?",
    "What do you think about the latest sports game?",
    "Can you send me the presentation from last week?"
]

# Generate 1,000 messages (mix of suspicious and non-suspicious)
data = []
for _ in range(500):  # 500 suspicious
    data.append([random.choice(suspicious_messages), 1])

for _ in range(500):  # 500 non-suspicious
    data.append([random.choice(non_suspicious_messages), 0])

# Shuffle the data
random.shuffle(data)

# Create a DataFrame
df = pd.DataFrame(data, columns=["message", "label"])

# Save to CSV file
df.to_csv('data/manual_labeled_messages.csv', index=False)

print("Dataset with 1000 messages has been generated and saved.")
