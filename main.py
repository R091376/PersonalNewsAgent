import os
import requests
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime
from langchain_classic.agents.agent import AgentExecutor
from langchain_classic.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#from dotenv import load_dotenv
#load_dotenv()


# This will look for variables in your system/GitHub environment
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check if keys are missing (useful for debugging GitHub Actions)
if not TELEGRAM_TOKEN or not CHAT_ID:
    raise ValueError("Missing environment variables! Check your GitHub Secrets.")

# 1. Initialize LLM
#llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
# Change your LLM to the 8B model
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# 2. RSS Tool optimized for CDATA in <description>
@tool(return_direct=True)
def fetch_indian_market_rss(query: str) -> str:
    """Extracts real-time news from Livemint and ET."""
    rss_urls = [
        "http://livemint.com/rss/markets",
        "https://economictimes.indiatimes.com/rssfeeds/1977021501.cms"
    ]
    
    news_items = []
    for url in rss_urls:
        try:
            feed = feedparser.parse(url)
            source = "Livemint" if "livemint" in url else "ET"
            for entry in feed.entries[:6]: # Increased depth
                desc = BeautifulSoup(entry.get('description', ''), "html.parser").get_text()
                news_items.append(
                    f"SOURCE: {source}\nTITLE: {entry.title}\nLINK: {entry.link}\nCONTENT: {desc[:300]}\n---"
                )
        except: continue

    return "\n".join(news_items) if news_items else "No current news found."

tools = [fetch_indian_market_rss]

# 3. Sector-Specific Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a Senior Market Strategist. Follow these steps exactly:\n"
        "1. Call fetch_market_news ONCE.\n"
        "2. Format the response as a 'MARKET INTELLIGENCE REPORT' for {date}.\n"
        "3. Group news into: 🏦 BANKING, 💻 IT & TECH, 📈 FII/MACRO.\n"
        "4. For EACH item, use this format: • <b>Title</b>: Short summary. <a href='LINK'>Read More</a>\n"
        "5. Use ONLY the info provided. If a sector is missing, skip it.\n"
        "6. End with 'Final Answer:'"
    )),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 4. Agent & Executor
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

# 5. Telegram & Execution
def send_to_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID, 
        "text": message, 
        "parse_mode": "HTML", 
        "disable_web_page_preview": True
    }
    requests.post(url, data=payload)

def run_agent():
    print(f"Running Factual Analysis at {datetime.now()}")
    try:
        # Ask for a very specific check to prevent generic summaries
        task = "Provide a detailed report on IT, Banking, and FII flows based ONLY on today's RSS data. Include links."
        current_date = datetime.now().strftime("%b %d, %Y")
        response = agent_executor.invoke({
            "input": task,
            "date": current_date  # This fills the {date} variable
        })
        
        # We wrap the output to ensure Telegram doesn't break
        final_report = f"🚀 <b>Live Market Intelligence</b>\n\n{response['output']}"
        send_to_telegram(final_report)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_agent()