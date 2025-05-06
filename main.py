import json
from typing import AsyncGenerator, Optional
import chainlit as cl
from chainlit.input_widget import Select
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from src.sample.graph import builder
from langchain_core.runnables.schema import StreamEvent
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import TavilySearchResults

from src.task1.graph import manual_agent
from src.task2_debate.graph import debate_agent


async def parse_stream_events(events: AsyncGenerator[StreamEvent, None]):
    async for event in events:
        type = event["event"]
        data = event["data"]
        node = event["metadata"].get("langgraph_node", "")

        # 跳过一些无用事件
        if type in [
            "on_chain_start",
            "on_chain_end",
            "on_chain_stream",
            "on_chat_model_start",
            "on_tool_start",
            "on_prompt_start",
            "on_prompt_end",
            "on_parser_start",
            "on_parser_end",
        ]:
            continue

        # 跳过空的chunk消息
        if type == "on_chat_model_stream" and (data["chunk"].content) == "":
            continue

        match type:
            case "on_chat_model_stream":

                result = {
                    "type": "chunk",
                    "data": data["chunk"].content,
                    "node": node,
                    "id": data["chunk"].id,
                }
                yield result

            case "on_chat_model_end":

                output = data["output"]
                tool_calls = output.tool_calls

                # 如果存在工具调用，则发送工具调用开始事件
                if len(tool_calls) > 0:
                    for tool_call in tool_calls:
                        result = {
                            "type": "tool_start",
                            "id": tool_call["id"],
                            "node": node,
                            "data": {
                                "name": tool_call["name"],
                                "args": tool_call["args"],
                            },
                        }
                        yield result
                # 否则发送AI消息
                else:
                    result = {
                        "type": "ai",
                        "data": output.content,
                        "id": output.id,
                        "node": node,
                    }
                    yield result

            case "on_tool_end":
                tool_call_id = data["output"].tool_call_id
                name = data["output"].name
                content = data["output"].content
                input = data["input"]
                result = {
                    "type": "tool_end",
                    "id": tool_call_id,
                    "node": node,
                    "data": {"content": content, "args": input, "name": name},
                }
                yield result

            case _:
                print(f"未处理的事件类型: {type}")
                print(data)


def format_json(message):
    if isinstance(message, str):
        try:
            return json.loads(message)
        except Exception:
            return message
    return message


graph_map = {
    "Sample Agent": builder.compile(checkpointer=MemorySaver()),
    "React Agent": create_react_agent(
        model=ChatOpenAI(
            model="gpt-4o-2024-11-20",
            temperature=0,
        ),
        tools=[
            TavilySearchResults(
                max_results=5,
                search_depth="advanced",
                include_answer=False,
                include_raw_content=False,
                include_images=False,
            )
        ],
        checkpointer=MemorySaver(),
    ),
    "Contact Assist": manual_agent,
    "Judge Assistant": debate_agent,
}


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    if (username, password) == ("root", "localpass"):
        return cl.User(identifier="admin", metadata={"role": "ADMIN"})
    else:
        return None


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Sample Agent",
            markdown_description="A demonstration agent that showcases the basic capabilities of LangGraph. Ask questions and see how the agent processes and responds.",
            icon="/public/search.png",
            starters=[
                cl.Starter(
                    label="今天黄金的价格是多少",
                    message="今天黄金的价格是多少",
                ),
            ],
        ),
        cl.ChatProfile(
            name="React Agent",
            markdown_description="A powerful ReAct agent built with GPT-4o and Tavily web search capabilities. It can research topics and answer queries by reasoning step-by-step.",
            icon="/public/search.png",
            starters=[
                cl.Starter(
                    label="今天黄金的价格是多少",
                    message="今天黄金的价格是多少",
                ),
            ],
        ),
        cl.ChatProfile(
            name="Contact Assist",
            markdown_description="A contact assistant that can help you find information about people and send emails.",
            icon="/public/assist.png",
            starters=[
                cl.Starter(
                    label="search person",
                    message="Please search for the person named Bob",
                ),
                cl.Starter(
                    label="send email for Jane Doe(only 1 person)",
                    message="Email Jane asking about the quote I sent yesterday. Thank her for choosing us.",
                ),
                cl.Starter(
                    label="send email for Johnson(more than 1 person)",
                    message="Email Johnson asking about the quote I sent yesterday. Thank him for choosing us.",
                ),
                cl.Starter(
                    label="send email for Jack(no person)",
                    message="Email Jack asking about the quote I sent yesterday. Thank him for choosing us.",
                ),
            ],
        ),
        cl.ChatProfile(
            name="Judge Assistant",
            markdown_description="一个辩论助手，帮助分析智能手机对青少年影响的正反方观点并给出总结。",
            starters=[
                cl.Starter(
                    label="开始辩论",
                    message="让我们开始讨论智能手机对青少年的影响。",
                ),
            ],
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    chat_profile = cl.user_session.get("chat_profile") or "Contact Assist"
    graph = graph_map[chat_profile]
    cl.user_session.set("graph", graph)

    await cl.ChatSettings(
        [
            Select(
                id="model",
                label="测试",
                values=["gpt-4o-2024-11-20", "gpt-4o-mini", "gpt-4.1-2025-04-14"],
                initial_index=0,
            ),
        ]
    ).send()


@cl.on_message
async def on_message(msg: cl.Message):
    session_id = cl.context.session.id
    graph: CompiledStateGraph = cl.user_session.get("graph")
    callbacks = [CallbackHandler(session_id=session_id)]
    config = {
        "configurable": {"thread_id": session_id},
        "callbacks": callbacks,
    }
    current_msg_id = None
    current_msg = None
    current_step: cl.Step | None = None

    events = graph.astream_events(
        {"messages": [HumanMessage(content=msg.content)]},
        config=config,
    )

    async for event in parse_stream_events(events):
        if event["type"] == "chunk":
            if current_msg_id is None:
                current_msg = cl.Message(content="")
                current_msg_id = event["id"]
            else:
                await current_msg.stream_token(event["data"])

        elif event["type"] == "ai":
            if current_msg is None:
                current_msg = cl.Message(content="", author="Assistant")
                await current_msg.send()
            current_msg.content = event["data"]
            await current_msg.send()
            current_msg = None
            current_msg_id = None

        elif event["type"] == "tool_start":
            tool_name = event["data"]["name"]
            tool_input = event["data"]["args"]
            async with cl.Step(name=tool_name, type="tool") as step:
                step.input = tool_input
            current_step = step

        elif event["type"] == "tool_end":
            tool_output = event["data"]["content"]
            print(tool_output)
            current_step.output = format_json(tool_output)
            await current_step.update()
            current_step = None
