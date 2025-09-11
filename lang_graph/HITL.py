# Human in the Loop (HITL)

from langchain.chat_models import init_chat_model
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def get_stock_price(symbol: str) -> float:
    '''Fetches the current stock price for the given symbol.
        :param symbol: The stock symbol to fetch the price for.
        :returns: The current stock price.
    '''
    return {
        "AAPL": 150.0,
        "GOOGL": 2800.0,
        "MSFT": 300.0,
        "AMZN": 3500.0,
        "TSLA": 700.0,
    }.get(symbol.upper(), 0.0)

@tool
def buy_stock(symbol: str, quantity: int, total_price) -> str:
    '''Buys a specified quantity of stock for the given symbol at the current price.'''
    descision = interrupt(f"Do you approve to buy {quantity} shares of {symbol} at total price ${total_price}? (yes/no)")

    if descision == "yes":
        return f"Successfully bought {quantity} shares of {symbol} at total price ${total_price}."  
    else:
        return "Purchase cancelled."


tools = [get_stock_price, buy_stock]
llm = init_chat_model("google_genai:gemini-2.0-flash")
llm_with_tools = llm.bind_tools(tools)

def chatbot_node(state: State) -> State:
    msg = llm_with_tools.invoke(state["messages"])
    return {"messages": [msg]}

memory = MemorySaver()
builder = StateGraph(State)
builder.add_node('chatbot', chatbot_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile(checkpointer=memory) 
config = {'configurable': {'thread_id': 1}}

#Step1: user askts the price
state = graph.invoke({"messages": [{"role": "user", "content" : "What is the latest price of 10 AMZN stock right now?"}]}, config=config)
print(state["messages"][-1].content)

#Step2: user decides to buy
state = graph.invoke({"messages": [{"role": "user", "content" : "Buy 10 AMZN at current price"}]}, config=config)
print(state.get('__interrupt__'))

descision = input("Approve (yes/no): ")

state = graph.invoke(Command(resume=descision), config=config)
print(state["messages"][-1].content)