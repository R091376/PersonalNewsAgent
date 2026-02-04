import os
import requests
import feedparser
import html
from bs4 import BeautifulSoup
from datetime import datetime
from langchain_classic.agents.agent import AgentExecutor
from langchain_classic.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#from dotenv import load_dotenv
#load_dotenv()

# Environment Variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not all([TELEGRAM_TOKEN, CHAT_ID, GROQ_API_KEY]):
    raise ValueError("Missing environment variables! Check GitHub Secrets for TELEGRAM_TOKEN, CHAT_ID, and GROQ_API_KEY.")

# 1. Helper: Safe HTML for Telegram
def safe_html(text: str) -> str:
    """Escapes characters that break Telegram HTML parsing."""
    return html.escape(str(text), quote=False)

# 2. Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# 3. RSS Tool with HTML Sanitization
@tool
def fetch_indian_market_rss(query: str) -> str:
    """Extracts real-time news from Livemint and ET. Only use this when asked."""
    rss_urls = [
        "http://livemint.com/rss/markets",
        "https://economictimes.indiatimes.com/rssfeeds/1977021501.cms"
    ]
    
    news_items = []
    for url in rss_urls:
        try:
            feed = feedparser.parse(url)
            source = "Livemint" if "livemint" in url else "ET"
            for entry in feed.entries[:6]:
                # 1. Clean description
                raw_desc = BeautifulSoup(entry.get('description', ''), "html.parser").get_text()
                clean_desc = " ".join(raw_desc.split())[:200]
                
                # 2. Escape HTML for safety
                s_title = safe_html(entry.title)
                s_desc = safe_html(clean_desc)
                s_link = entry.link # Links don't need escaping in <a href>
                
                news_items.append(
                    f"SOURCE: {source}\nTITLE: {s_title}\nLINK: {s_link}\nCONTENT: {s_desc}\n---"
                )
        except Exception: 
            continue

    return "\n".join(news_items) if news_items else "No current news found."

tools = [fetch_indian_market_rss]

# 4. Prompt optimized for Spacing and Stability
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a Senior Market Strategist.\n"
        "STEPS:\n"
        "1. Use fetch_indian_market_rss once.\n"
        "2. Organize results into: 🏦 BANKING, 💻 IT & TECH, 📈 FII/MACRO.\n"
        "3. For EACH story, use this EXACT format with DOUBLE SPACING between items:\n"
        "• <b>Title</b>: Summary. <a href='LINK'>Read More</a>\n\n"
        "4. If no news for a sector, skip it. If no news at all, say 'No news found.'\n"
        "5. Final Answer: Your formatted report."
    )),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 5. Agent Setup
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

# 6. Telegram Logic
def send_to_telegram(message):
    # IMPORTANT: `message` is expected to be HTML-safe for Telegram.
    # Use `safe_html()` on any user-provided or scraped text fields to
    # avoid breaking Telegram's HTML parsing while keeping intended
    # formatting (e.g., <b>, <a>) intact.
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID, 
        "text": message, 
        "parse_mode": "HTML", 
        "disable_web_page_preview": True
    }
    try:
        r = requests.post(url, data=payload)
        if r.status_code != 200:
            print(f"❌ Telegram Error {r.status_code}: {r.text}")
        else:
            print("✅ Message sent successfully!")
    except Exception as e:
        print(f"❌ Connection Error: {e}")

def run_agent():
    current_date = datetime.now().strftime("%b %d, %Y")
    print(f"🚀 Executing Report for {current_date}")
    
    try:
        task = "Analyze today's RSS data for IT, Banking, and FII flows. Format into a clean report with links."
        response = agent_executor.invoke({"input": task, "date": current_date})
        
        # Clean up the "Final Answer:" tag if the LLM includes it
        raw_output = response.get('output', '')
        clean_output = raw_output.replace("Final Answer:", "").strip()

        if not clean_output or "no news" in clean_output.lower():
            report = f"📊 <b>Market Update</b>\n{current_date}\n\n⚠️ No new items found in RSS feeds."
        else:
            report = f"📊 <b>Market Intelligence Report</b>\n{current_date}\n\n{clean_output}"
        
        send_to_telegram(report)

    except Exception as e:
        # If the code crashes, tell you via Telegram
        err_detail = safe_html(str(e))[:300]
        error_msg = f"❌ <b>System Error</b>\n{current_date}\n\n<code>{err_detail}</code>"
        send_to_telegram(error_msg)

if __name__ == "__main__":
    run_agent()