# 📊 Khasim's Personal News Bot

An automated AI-powered Telegram bot that delivers curated market intelligence and financial news directly to your inbox. The bot intelligently aggregates and analyzes real-time market data from Indian financial news sources and sends formatted reports to Telegram.

## 🎯 What It Does

This bot:
- **Fetches Real-Time News**: Pulls the latest market news from [Livemint](https://www.livemint.com) and [Economic Times](https://economictimes.indiatimes.com)
- **Intelligent Curation**: Uses AI (Llama 3.3 70B via Groq) to analyze and organize news into categories:
  - 🏦 Banking & Finance
  - 💻 IT & Technology  
  - 📈 FII Flows & Macro
- **Automated Delivery**: Sends a beautifully formatted daily report to your Telegram chat
- **Error Handling**: Notifies you immediately if anything goes wrong

## 📸 Sample Output

Here's how the report looks in Telegram:

![Market News Report - Example 1](screenshots/news_bot_1.png)

![Market News Report - Example 2](screenshots/news_bot_2.png)

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A Telegram account
- A Groq API key (free at [console.groq.com](https://console.groq.com))

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/khasimkhan/market_news_agent.git
   cd market_news_agent
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```
   TELEGRAM_TOKEN=your_telegram_bot_token_here
   CHAT_ID=your_telegram_chat_id_here
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Get your Telegram credentials**
   - **Bot Token**: Create a bot via [@BotFather](https://t.me/botfather) on Telegram
   - **Chat ID**: Send a message to your bot, then visit `https://api.telegram.org/bot<TOKEN>/getUpdates` to find your chat ID

6. **Run the bot**
   ```bash
   python main.py
   ```

## 📋 How It Works

1. The bot initializes a LangChain agent with the Llama 3.3 70B model
2. It fetches the latest market news from RSS feeds of Livemint and Economic Times
3. The AI agent organizes the news into relevant financial categories
4. It formats the news with proper HTML styling for Telegram
5. Sends the formatted report to your Telegram chat
6. If any errors occur, you get notified via Telegram

## 🔧 Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `TELEGRAM_TOKEN` | Your Telegram bot token | `123456789:ABCDEFGHIJKLMNOPqrstuvwxyz...` |
| `CHAT_ID` | Your Telegram chat/user ID | `987654321` |
| `GROQ_API_KEY` | Your Groq API key for LLM | `gsk_xxxxxxxxxxxxx...` |

### Scheduled Execution (GitHub Actions)

The bot runs automatically on a schedule using GitHub Actions. Check [`.github/workflows/run_bot.yml`](.github/workflows/run_bot.yml) to modify the schedule.

Current schedule: Runs daily at a specified time (configurable in the workflow file)

## 📦 Dependencies

- **feedparser**: Parse RSS feeds
- **beautifulsoup4**: Clean HTML content from news
- **langchain-groq**: Integration with Groq's Llama model
- **langchain_classic**: LangChain tools and agents
- **langchain_core**: Core LangChain components
- **requests**: HTTP requests for Telegram API
- **python-dotenv**: Load environment variables from `.env`

## 🛠️ Customization

### Change News Sources
Edit the `fetch_indian_market_rss()` function in `main.py`:
```python
rss_urls = [
    "http://livemint.com/rss/markets",
    "https://economictimes.indiatimes.com/rssfeeds/1977021501.cms",
    # Add more RSS feeds here
]
```

### Modify Report Categories
Update the system prompt in the `prompt` variable:
```python
"2. Organize results into: 🏦 BANKING, 💻 IT & TECH, 📈 FII/MACRO.\n"
```

### Change AI Model
Update the model in `main.py`:
```python
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
```

## 📊 Current Limitations

- **Single Timezone**: Currently hardcoded for IST (India Standard Time). No timezone flexibility
- **Indian Markets Only**: Focused exclusively on Indian financial news sources (Livemint, ET)
- **Manual Local Setup**: Requires manual environment variable configuration for local runs
- **No Database/History**: Doesn't store historical news or reports - each run is independent
- **Limited News Sources**: Only 2 RSS feeds. Doesn't include other financial sources or APIs
- **No Error Retry Logic**: If a news fetch fails, it doesn't retry or fallback gracefully
- **No User Customization**: All users get the same categories and formatting
- **Telegram Only**: Output is limited to Telegram - no other notification channels
- **No Rate Limiting**: Could hit API rate limits if run too frequently
- **No Content Filtering**: Sometimes publishes promotional or low-quality content without filtering
- **Static Categories**: Fixed category organization doesn't adapt to market conditions

## 🔮 Future Upgrades

### Short Term
- [ ] **Add `.env.example`**: Template file for easier setup
- [ ] **Database Integration**: Store news history in SQLite/PostgreSQL to avoid duplicates
- [ ] **Retry Logic**: Implement exponential backoff for failed API calls
- [ ] **News Deduplication**: Skip duplicate articles across days
- [ ] **User Configuration**: Allow users to customize categories and filters via Telegram commands

### Medium Term
- [ ] **Multi-Language Support**: Translate summaries to English/other languages
- [ ] **Sentiment Analysis**: Add market sentiment indicators to news summaries
- [ ] **Multiple Notification Channels**: Discord, Slack, Email integrations
- [ ] **Web Dashboard**: Simple website to view historical reports
- [ ] **More News Sources**: Add international news, crypto, commodities sources
- [ ] **Configurable Timezones**: Support multiple timezone deliveries

### Long Term
- [ ] **Advanced NLP**: Extract entities (companies, people, events) from news
- [ ] **Predictive Analytics**: Correlate news with stock price movements
- [ ] **Multi-User Support**: Different reports for different user profiles
- [ ] **Mobile App**: Dedicated mobile app for reading reports
- [ ] **Social Media Integration**: Auto-post summaries to Twitter/LinkedIn
- [ ] **Interactive Bot**: Telegram commands to ask follow-up questions about news
- [ ] **Price Tracking**: Integrate with stock APIs to show price impact of news

## ⚠️ Troubleshooting

### Bot doesn't send messages
- Verify `TELEGRAM_TOKEN` and `CHAT_ID` are correct
- Check if your bot has permission to send messages in that chat
- Ensure the bot is not blocked

### Missing news in reports
- Check if RSS feeds are working by visiting them in a browser
- Verify internet connection
- Check Groq API quota

### Import errors
- Make sure all packages from `requirements.txt` are installed
- Verify Python version is 3.10+

## 📄 License

Personal project. Feel free to use and modify for your own purposes.

## 🤝 Contributing

This is a personal project, but suggestions and improvements are welcome!

---

**Stay informed about markets. Let the bot do the work! 📈**
