from getpass import getpass
import os
from typing import Optional
from fastapi import Depends, HTTPException
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END


class LangChainService:
    def __init__(self):
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        
        os.environ["LANGSMITH_TRACING"] = "true"
        if not os.environ.get("LANGSMITH_API_KEY"):
            raise ValueError("LANGSMITH_API_KEY environment variable is not set.")

        self.llm = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        graph_builder = StateGraph(MessagesState)
        
        # Define the retrieve tool
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information related to a query."""
            DUMMY_DOCS = [
                {
                    "page_content": "This is a dummy document content. TEST RUN. TEST RUN.",
                    "metadata": {"source": "dummy_source_1"},
                },
                {
                    "page_content": "This is another dummy document content. TEST RUN. TEST RUN.",
                    "metadata": {"source": "dummy_source_2"},
                },
            ]
            serialized = "\n\n".join(
                (f"Source: {doc['metadata']}\nContent: {doc['page_content']}")
                for doc in DUMMY_DOCS
            )
            return serialized, DUMMY_DOCS
        
        # Define functions that capture self in closure
        def query_or_respond(state: MessagesState):
            """Generate tool call for retrieval or respond."""
            llm_with_tools = self.llm.bind_tools([retrieve])
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}
        
        def generate(state: MessagesState):
            """Generate answer."""
            # Get generated ToolMessages
            recent_tool_messages = []
            for message in reversed(state["messages"]):
                if message.type == "tool":
                    recent_tool_messages.append(message)
                else:
                    break
            tool_messages = recent_tool_messages[::-1]

            # Format into prompt
            docs_content = "\n\n".join(doc.content for doc in tool_messages)
            system_message_content = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                f"{docs_content}"
            )
            conversation_messages = [
                message
                for message in state["messages"]
                if message.type in ("human", "system")
                or (message.type == "ai" and not message.tool_calls)
            ]
            prompt = [SystemMessage(system_message_content)] + conversation_messages

            # Run
            response = self.llm.invoke(prompt)
            return {"messages": [response]}
        
        # Create tools node
        tools = ToolNode([retrieve])
        
        # Add nodes - now using local functions that have access to self
        graph_builder.add_node("query_or_respond", query_or_respond)
        graph_builder.add_node("tools", tools)
        graph_builder.add_node("generate", generate)
        
        # Set up edges
        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)
        
        return graph_builder.compile()
    
    async def process_query(self, messages: list):
        """Process a query through the graph."""
        try:
            result = await self.graph.ainvoke(
                input={"messages": messages},
                stream_mode="values",
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# Singleton instance
_langchain_service: Optional[LangChainService] = None


def get_langchain_service() -> LangChainService:
    """Dependency to get the LangChain service instance."""
    global _langchain_service
    if _langchain_service is None:
        _langchain_service = LangChainService()
    return _langchain_service