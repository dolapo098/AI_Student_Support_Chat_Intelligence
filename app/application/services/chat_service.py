import logging
import os
import time
from typing import Dict, List

from app.contracts.dtos.chat_dtos import (
    ChatMessageDto,
    ChatRequest,
    ChatResponse,
    ClearSessionResponse,
)
from app.contracts.providers.i_knowledge_provider import IKnowledgeProvider
from app.contracts.providers.i_llm_provider import ILLMProvider
from app.contracts.services.i_chat_service import IChatService
from app.domain.exceptions.chat_exception import InvalidChatRequestException
from app.infrastructure.observability.conversation_audit import append_conversation_audit_line
from app.infrastructure.security.pii_scrubber import PIIScrubber

logger = logging.getLogger(__name__)

_KENT_SYSTEM_PROMPT = """
You are Kay - **Kent Advice for You**, the friendly AI student-support assistant for the
University of Kent. Sound like a warm, approachable adviser: conversational, encouraging, clear -
not stiff or robotic. At most **one** light emoji per reply when it fits (e.g. 👋 or 😊).

**Intent disambiguation (vague or broad messages):**
When the user is vague ("help", "connect me", "I need someone", "support", "what about courses")
**and** retrieved context does not yet support a precise answer, **do not guess**. Briefly acknowledge
them, then offer **3-5 short bullet options** so they can steer you - for example:
- **Applying & programmes** - entry, courses (www.kent.ac.uk/courses)
- **Fees & funding**
- **Study & deadlines** - assessments, extenuating circumstances
- **Wellbeing**
- **Housing / campus life**
- **Speaking to a person** - Student Services or their **school office**
End with **one** friendly clarifying question (e.g. prospective vs current student, or which area
matters most). Keep this pattern **one turn**; avoid a long interview unless they stay vague.

**Progressive narrowing:**
For broad topics (postgraduate, "accounting", "which degree"), ask for **missing basics** before
detailed facts: level (UG/PG), rough subject area, or what they want next (how to apply vs module
detail). Only state programme specifics, deadlines, or contacts that appear in the **context** or
are generic Kent signposting you were given below - never invent programme names or requirements.

**Capability list** (only when asked "what can you do", capabilities, "how do you work"):
Give a structured, **Kent-specific** rundown with bold headings and bullets - admissions &
programmes, student life & funding, study & deadlines, wellbeing, getting to a human. Close with one
question to learn if they're applying or already at Kent.

**How you get information** (scraping, training, sources):
You **do not** browse the live web. Answers are grounded in **official Kent content** prepared for
this assistant (passages in the context below). If something isn't in context, say so honestly and
signpost Student Services or the right kent.ac.uk page.

**Using the context block:**
Prioritise facts from excerpts; quote numbers or policy lines only when they appear there. If
context is thin, stay friendly, suggest next steps, and avoid dumping the full contact block every
time.

**Rapport** (thanks, compliments, "you're good", light chat):
Reply in **one or two short sentences** - warm acknowledgment, then invite a Kent question. Do **not**
repeat your full capability list unless they ask again.

**Off-topic:**
Politely redirect to University of Kent support with 2-3 example topics.

**Safety & accuracy:**
Never invent deadlines, fees, grades, or staff names not in the context. For wellbeing distress,
empathy first; signpost Kent Wellbeing / Nightline / Samaritans. No politics, medical diagnosis, or
long unrelated tangents.

**Getting to a human:**
Student Services: student-enquiries@kent.ac.uk | 01227 764000

**Wellbeing signposting** (emotional support, not only crisis):
Kent Student Wellbeing: 01227 823158 | wellbeing@kent.ac.uk
Nightline (student-run, term-time evenings): 01227 769823
Samaritans (24/7): 116 123
""".strip()

_WELLBEING_RESPONSE_SUFFIX = (
    "\n\n---\n"
    "Your wellbeing matters. If you are struggling, please reach out:\n"
    "- **Kent Student Wellbeing**: 01227 823158 | wellbeing@kent.ac.uk\n"
    "- **Nightline** (evenings, term-time): 01227 769823\n"
    "- **Samaritans** (24/7, free): 116 123\n"
    "You are not alone."
)

_SUGGESTED_TOPICS = [
    "What can Kay help me with?",
    "How do you get your information?",
    "I need to speak to someone - who should I contact?",
    "How do I apply to Kent?",
    "What wellbeing support is available?",
]


class ChatService(IChatService):
    """
    Summary: Orchestrates RAG pipeline - retrieves Kent context, detects wellbeing,
    and generates grounded responses via the configured LLM provider.
    """

    def __init__(
        self,
        llm_provider: ILLMProvider,
        knowledge_provider: IKnowledgeProvider,
        pii_scrubber: PIIScrubber,
    ):
        self._llm = llm_provider
        self._knowledge = knowledge_provider
        self._scrub = pii_scrubber
        self._sessions: Dict[str, List[ChatMessageDto]] = {}

    async def chat(self, request: ChatRequest) -> ChatResponse:

        if not request.message.strip():
            raise InvalidChatRequestException("Message cannot be empty.")

        # Server-side store only (session_id): client-supplied history is not merged -
        # avoids desync when the API does not echo history back in ChatResponse.
        session_history = list(self._sessions.get(request.session_id, []))

        t0 = time.perf_counter()
        is_wellbeing = await self._llm.detect_wellbeing(request.message)
        t1 = time.perf_counter()

        context_hits = self._knowledge.search(request.message, top_k=5)
        t2 = time.perf_counter()
        context_chunks = [h.text for h in context_hits]
        retrieved_chunk_indices = [h.chunk_index for h in context_hits]
        context = "\n\n".join(context_chunks) if context_chunks else (
            "No specific Kent University document context available for this query."
        )

        llm_message = self._scrub.for_llm(request.message)
        llm_history = [
            ChatMessageDto(
                role=m.role,
                content=self._scrub.for_llm(m.content),
                timestamp=m.timestamp,
            )
            for m in session_history
        ]

        answer = await self._llm.generate_response(
            message=llm_message,
            context=context,
            history=llm_history,
            system_prompt=_KENT_SYSTEM_PROMPT,
        )
        t3 = time.perf_counter()

        ms_wellbeing = round((t1 - t0) * 1000.0, 2)
        ms_retrieval = round((t2 - t1) * 1000.0, 2)
        ms_llm = round((t3 - t2) * 1000.0, 2)
        ms_total = round((t3 - t0) * 1000.0, 2)

        if is_wellbeing:
            answer += _WELLBEING_RESPONSE_SUFFIX

        session_history.append(
            ChatMessageDto(role="user", content=request.message)
        )
        session_history.append(
            ChatMessageDto(role="assistant", content=answer)
        )

        self._sessions[request.session_id] = session_history[-20:]

        max_audit_resp = int(os.getenv("CONVERSATION_AUDIT_MAX_RESPONSE_CHARS", "8000") or "8000")
        scrubbed_answer = self._scrub.for_logs(answer)
        if len(scrubbed_answer) > max_audit_resp:
            scrubbed_answer = scrubbed_answer[:max_audit_resp] + "…[truncated]"

        append_conversation_audit_line(
            {
                "session_id": request.session_id,
                "scrubbed_user_message": self._scrub.for_logs(request.message),
                "scrubbed_assistant_message": scrubbed_answer,
                "retrieved_chunk_indices": retrieved_chunk_indices,
                "retrieved_chunk_count": len(retrieved_chunk_indices),
                "context_empty": len(retrieved_chunk_indices) == 0,
                "wellbeing_flag": is_wellbeing,
                "latency_ms_wellbeing": ms_wellbeing,
                "latency_ms_retrieval": ms_retrieval,
                "latency_ms_llm": ms_llm,
                "latency_ms_total": ms_total,
            }
        )

        logger.info(
            "Chat processed - session: %s | wellbeing: %s | chunks: %d | retrieval_ms: %s | "
            "llm_ms: %s | message_snippet: %.120s",
            request.session_id,
            is_wellbeing,
            len(context_chunks),
            ms_retrieval,
            ms_llm,
            self._scrub.for_logs(request.message),
        )

        return ChatResponse(
            answer=answer,
            session_id=request.session_id,
            is_wellbeing=is_wellbeing,
            suggested_topics=_SUGGESTED_TOPICS if len(session_history) <= 2 else [],
        )

    def clear_session(self, session_id: str) -> ClearSessionResponse:

        if session_id in self._sessions:
            del self._sessions[session_id]

        return ClearSessionResponse(
            message="Session cleared successfully.",
            session_id=session_id,
        )
