import asyncio
import telegram

# token required
lines = open('.bot_token', 'rt').readlines()
TOKEN = lines[0].strip()
MYCHAT_ID = lines[1].strip()

async def send_message(text):
    bot = telegram.Bot(TOKEN)
    async with bot:
        print(await bot.send_message(text=text, chat_id=MYCHAT_ID))

def send_to_mybot(text="my job is done!"):
    asyncio.run(send_message(text))