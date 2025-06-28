import json
import os
from typing import Annotated

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class State(TypedDict):
    messages: Annotated[
        list[BaseMessage, AIMessage, HumanMessage, ToolMessage], add_messages
    ]


llm = init_chat_model("openai:gpt-4o-mini")
tool = TavilySearch(max_results=2)
graph_builder = StateGraph(State)

tools = [tool]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")


# class BasicToolNode:
#     """A node that runs the tools requested in the last Message."""

#     def __init__(self, tools: list) -> None:
#         self.tools_by_name = {tool.name: tool for tool in tools}

#     def __call__(self, state: State):
#         message = state["messages"][-1]

#         tools_result = []

#         for tool_call in message.tool_calls:
#             tool_result = self.tools_by_name[tool_call["name"]].invoke(
#                 tool_call["args"]
#             )

#             tools_result.append(
#                 ToolMessage(
#                     content=json.dumps(tool_result),
#                     name=tool_call["name"],
#                     tool_call_id=tool_call["id"],
#                 )
#             )

#         return {"messages": tools_result}


# tool_node = BasicToolNode(tools=[tool])
# graph_builder.add_node("tools", tool_node)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)


# def route_tools(
#     state: State,
# ):
#     message = state["messages"][-1]

#     if hasattr(message, "tool_calls") and len(message.tool_calls) > 0:
#         return "tools"

#     return END


# graph_builder.add_conditional_edges(
#     "chatbot",
#     route_tools,
#     {"tools": "tools", END: END},
# )

graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile()
# display(Image(graph.get_graph().draw_mermaid_png()))
print(graph.get_graph().draw_ascii())


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [HumanMessage(user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


if __name__ == "__main__":
    while True:
        user_input = input("User: ")

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
