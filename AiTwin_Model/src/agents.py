# agents.py

from typing import List, Tuple, Optional, Any
from autogen import ConversableAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from rag_system import RAGSystem # Import the new RAGSystem class

class RAGAssistantAgent(ConversableAgent):
    """A ConversableAgent subclass that uses the RAGSystem to retrieve and answer queries."""
    def __init__(self, rag_system: RAGSystem, **kwargs):
        super().__init__(**kwargs)
        self.rag_system = rag_system

    async def on_messages(self, messages: List[TextMessage], cancellation_token: CancellationToken, user_id: str, peer_id: Optional[str] = None) -> Tuple[TextMessage, dict]:
        """Processes incoming messages using the RAG pipeline."""
        rag_response, metrics = self.rag_system.retrieve_and_answer(messages, user_id, peer_id)
        return TextMessage(content=rag_response, source=self.name), metrics

async def get_agent(history: list[dict[str, Any]], rag_system: RAGSystem) -> RAGAssistantAgent:
    """Factory function to create and initialize the RAG assistant agent."""
    agent = RAGAssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        rag_system=rag_system
    )
    # Safely convert history dictionaries to TextMessage objects
    agent._history = [TextMessage(**msg) for msg in history if 'source' in msg and 'content' in msg]
    return agent