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
from langgraph.checkpoint.memory import MemorySaver
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


class LangChainService:
    def __init__(self):
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        
        os.environ["LANGSMITH_TRACING"] = "true"
        if not os.environ.get("LANGSMITH_API_KEY"):
            raise ValueError("LANGSMITH_API_KEY environment variable is not set.")
        
        if not os.environ.get("QDRANT_API_KEY"):
            raise ValueError("QDRANT_API_KEY environment variable is not set.")
        
        if not os.environ.get("QDRANT_API_ENDPOINT"):
            raise ValueError("QDRANT_API_ENDPOINT environment variable is not set.")

        QDRANT_COLLECTION_NAME = "subjects"

        client = QdrantClient(
            url=os.environ["QDRANT_API_ENDPOINT"],
            api_key=os.environ["QDRANT_API_KEY"]
        )

        embeddings = OllamaEmbeddings(
            model="bge-m3",
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434"),
        )

        self.llm = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")
        self.graph = self._build_graph()
        self.vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=QDRANT_COLLECTION_NAME,
            url=os.environ["QDRANT_API_ENDPOINT"],
            api_key=os.environ["QDRANT_API_KEY"],
        )

    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        graph_builder = StateGraph(MessagesState)
        
        # Define the retrieve tool
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information related to a query."""

            query_embedding = self.vector_store.embeddings.embed_query(query)
            
            # Query Qdrant directly
            search_result = self.vector_store.client.search(
                collection_name="subjects",
                query_vector=query_embedding,
                limit=2,
                with_payload=True
            )
            
            # Map your actual field names to page_content
            retrieved_docs = []
            for point in search_result:
                payload = point.payload

                page_content = payload.get('subject_name', '') + " "+ payload.get('subject_content', '') + " " + payload.get('subject_effects', '')

                doc = Document(
                    page_content=page_content,
                    metadata=payload
                )
                retrieved_docs.append(doc)

                serialized = "\n\n".join(
                    (f"\nSubject content: {doc.page_content}\n")
                    for doc in retrieved_docs
                )
            return serialized, retrieved_docs
      
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
                "the question. If you don't know the answer, or the document contents are empty, say that you "
                "don't know, or that the documents don't provide the information. Use three sentences maximum and keep the "
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

        memory = MemorySaver()
        
        return graph_builder.compile(checkpointer=memory)
    
    async def process_query(self, messages: list):
        """Process a query through the graph."""

        config = {"configurable": {"thread_id": "abc123"}}
        try:
            result = await self.graph.ainvoke(
                input={"messages": messages},
                stream_mode="values",
                config=config
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