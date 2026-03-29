"""
KnowledgeBaseBuilder - builds the FAISS vector index from Kent University content.

This script is run once before starting the server (or automatically on startup
via the FastAPI lifespan event in app/main.py).

    python scripts/build_knowledge_base.py
"""

import logging
import pickle
from pathlib import Path
from typing import List

import numpy as np
import faiss
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


_KENT_PAGES = [
    ("Student Support", "https://www.kent.ac.uk/student-support"),
    ("Contact Us", "https://www.kent.ac.uk/contact-us"),
    ("International Deadlines", "https://www.kent.ac.uk/international/international-admission-deadlines"),
    ("Assessment", "https://www.kent.ac.uk/student-guide/assessment"),
    ("Library", "https://www.kent.ac.uk/library"),
    ("Accommodation", "https://www.kent.ac.uk/accommodation"),
    ("Careers", "https://www.kent.ac.uk/careers"),
    ("IT Services", "https://www.kent.ac.uk/it"),
]

_STATIC_KNOWLEDGE = [
    """
    University of Kent verified contact directory:
    - Main switchboard: +44 (0)1227 764000
    - Admissions (Future Students): study@kent.ac.uk | +44 (0)1227 768896
    - Student Support & Wellbeing (Canterbury): KentSSW@kent.ac.uk | +44 (0)1227 823158
    - Student Support & Wellbeing (Medway): MedwaySSW@kent.ac.uk | +44 (0)1634 888474
    - Accommodation: accomm@kent.ac.uk | +44 (0)1227 766660
    - Careers & Employability: careerhelp@kent.ac.uk | +44 (0)1227 823299
    - IT Support (Helpdesk): helpdesk@kent.ac.uk | +44 (0)1227 824888
    - Student Fees: +44 (0)1227 824242
    - Financial Hardship: financialhardship@kent.ac.uk
    - Campus Security (Emergencies): +44 (0)1227 823333
    - Campus Security (Non-emergencies): +44 (0)1227 823300
    - Nexus (Student Help): nexus@kent.ac.uk | +44 (0)1227 827932
    - Kent Students Union: kentunion@kent.ac.uk | +44 (0)1227 824200
    - Templeman Library: +44 (0)1227 824777
    """,
    """
    University of Kent international admission deadlines 2026:
    April 2026 intake:
    - 13 April 2026: Term start date
    - 30 March 2026: Last date for deposit payment
    - 7 April 2026: CAS issuance deadline
    September 2026 intake:
    - 26 September 2026: September term start date
    - 24 August 2026: Application deadline for undergraduate and postgraduate taught courses
    - 4 September 2026: Last date for applicants to meet all conditions and accept offer
    - 11 September 2026: CAS issuance deadline
    - 29 June 2026: Application deadline for postgraduate research courses
    """,
    """
    Assessment and exam information at University of Kent:
    - Assignment submission: via Moodle (the virtual learning environment)
    - Exam timetables published in My Kent Uni portal
    - Extenuating circumstances (EC) forms submitted before the assessment deadline
    - Mitigating circumstances: contact your School Office
    - Results: available on My Kent Uni portal after boards
    - Resits: typically in August - check your school's guidelines
    - Grade appeal process: via the Academic Appeals procedure on kent.ac.uk
    """,
    """
    University of Kent Student Wellbeing Support:
    - Student Wellbeing Service offers free, confidential counselling and mental health support
    - Drop-in sessions available at Keynes College (Canterbury campus)
    - Online booking at: kent.ac.uk/student-support-and-wellbeing/wellbeing
    - Crisis support: contact wellbeing team or call Samaritans on 116 123 (free, 24/7)
    - Togetherall (online peer support): free access for all Kent students
    - Student Minds: student mental health charity resources
    """,
    """
    University of Kent Library services:
    - Canterbury campus: Templeman Library - open 24/7 during term time
    - Medway campus: Drill Hall Library
    - Resources: e-books, journals, databases available via kent.ac.uk/library
    - Book borrowing: use Library Search to find and reserve books
    - Study spaces: bookable via the Library website
    - IT help desk located in Templeman Library, ground floor
    """,
    """
    Admissions at University of Kent:
    - Undergraduate: apply via UCAS (www.ucas.com)
    - Postgraduate: apply directly via kent.ac.uk/courses
    - International students: same process, additional documents required (passport, qualifications)
    - Entry requirements: vary per course - check individual course pages
    - Clearing: available for eligible courses in August
    - Widening Access schemes: speak to Student Recruitment team
    - Admissions contact: admissions@kent.ac.uk | 01227 827272
    """,
]


class KnowledgeBaseBuilder:
    """
    Builds and persists the FAISS vector index from Kent University content.

    Responsibilities:
      - Scrape Kent University web pages
      - Split raw text into overlapping chunks
      - Generate local embeddings via sentence-transformers
      - Save FAISS index and chunk list to disk
    """

    def __init__(
        self,
        index_dir: str = "faiss_index",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        self._index_dir = Path(index_dir)
        self._chunks: List[str] = []

        logger.info("Loading embedding model: %s", embedding_model)
        self._embedder = SentenceTransformer(embedding_model)

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def scrape_pages(self, pages: List[tuple]) -> List[str]:
        """
        Scrapes a list of (label, url) pairs and returns cleaned text blocks.
        Failed pages are skipped with a warning - never blocks the build.
        """

        scraped: List[str] = []

        for label, url in pages:
            try:
                headers = {"User-Agent": "KentAI-KnowledgeBuilder/1.0"}
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")

                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()

                main = soup.find("main") or soup.find("article") or soup.find("body")
                text = main.get_text(separator=" ", strip=True) if main else ""

                if text:
                    logger.info("Scraped [%s] - %d chars", label, len(text))
                    scraped.append(f"[{label}]\n{text}")

            except Exception as exc:
                logger.warning("Could not scrape [%s] %s - skipping: %s", label, url, exc)

        return scraped

    def chunk_texts(self, raw_texts: List[str]) -> List[str]:
        """
        Splits raw text blocks into overlapping chunks for embedding.
        """

        chunks: List[str] = []

        for text in raw_texts:
            chunks.extend(self._splitter.split_text(text))

        logger.info("Total chunks after splitting: %d", len(chunks))

        self._chunks = chunks
        return chunks

    def build_index(self) -> faiss.Index:
        """
        Generates sentence embeddings for all chunks and builds a FAISS L2 index.
        """

        if not self._chunks:
            raise ValueError("No chunks to index - call chunk_texts() first.")

        logger.info("Generating embeddings for %d chunks...", len(self._chunks))

        vectors = self._embedder.encode(
            self._chunks,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        vectors = np.array(vectors, dtype="float32")

        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)

        logger.info(
            "FAISS index built - %d vectors, %d dims.",
            index.ntotal, vectors.shape[1]
        )

        return index

    def save(self, index: faiss.Index) -> None:
        """
        Persists the FAISS index and chunk list to disk.
        """

        self._index_dir.mkdir(exist_ok=True)

        faiss.write_index(index, str(self._index_dir / "index.faiss"))

        with open(self._index_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

        logger.info(
            "Knowledge base saved to %s - %d chunks ready.",
            self._index_dir, len(self._chunks)
        )

    def run(
        self,
        static_knowledge: List[str],
        pages: List[tuple],
    ) -> None:
        """
        High-level orchestration: scrape → chunk → embed → save.
        """

        logger.info("Starting Kent knowledge base build...")

        raw_texts = list(static_knowledge)
        raw_texts.extend(self.scrape_pages(pages))

        self.chunk_texts(raw_texts)

        index = self.build_index()

        self.save(index)

        logger.info("Knowledge base build complete.")


if __name__ == "__main__":
    builder = KnowledgeBaseBuilder(
        index_dir="faiss_index",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=800,
        chunk_overlap=150,
    )

    builder.run(
        static_knowledge=_STATIC_KNOWLEDGE,
        pages=_KENT_PAGES,
    )
