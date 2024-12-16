# bot.py
from telethon import TelegramClient, events
import joblib
from config import api_id, api_hash, bot_token

# Load the trained model and vectorizer
model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# Initialize Telegram bot
bot = TelegramClient('bot_session', api_id, api_hash).start(bot_token=bot_token)

# Function to classify messages
def classify_message(message):
    # Vectorize the message
    message_vectorized = vectorizer.transform([message])
    
    # Predict using the model
    prediction = model.predict(message_vectorized)
    
    # If prediction is 1 (suspicious), return a warning
    return 'Suspicious' if prediction == 1 else 'Not Suspicious'

# Event handler for new messages
@bot.on(events.NewMessage)
async def handle_message(event):
    print(f"Received message: {event.message.text}")  # Debugging message
    
    message = event.message.message

    # Classify the message
    classification = classify_message(message)
    
    print(f"Message classified as: {classification}")  # Debugging classification

    if classification == 'Suspicious':
        await event.reply("⚠️ Suspicious message detected! Please be cautious.")
        print(f"Flagged message: {message}")

# Run the bot
print("Bot is running...")
bot.run_until_disconnected()
