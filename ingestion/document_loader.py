"""
ingestion/document_loader.py

Loads PDF documents from disk and returns LangChain Document objects
with rich metadata attached. This is the entry point for all 20 research
papers before any chunking strategy is applied.

TEACHING NOTE:
- We use pypdf directly (not LangChain's PyPDFLoader) because we need
  per-page error isolation — if page 47 of GPT-3 is corrupted, we log it
  and continue loading pages 1-46 and 48-75.
- Metadata attached here (source, page, paper_id) flows through chunking
  into the FAISS index and is returned with every retrieval result.
  This is how RAGAS knows which paper a chunk came from.

PROD SCALE (20,000 docs / 800K pages):
- Replace synchronous loader with async batch loader using asyncio.gather
- Add deduplication by content hash before indexing
- Stream documents into chunker rather than loading all into memory
"""

from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from loguru import logger
from pypdf import PdfReader
from pypdf.errors import PdfReadError

PAPERS_DIR = Path(__file__).parent.parent / "data" / "papers"

# Paper ID → human-readable title mapping
# Used to populate metadata on every chunk for interpretable eval results
PAPER_TITLES = {
    "01_attention_is_all_you_need": "Attention Is All You Need",
    "02_bert": "BERT",
    "03_gpt3": "GPT-3",
    "04_llama": "LLaMA",
    "05_llama2": "LLaMA 2",
    "06_rag": "Retrieval-Augmented Generation",
    "07_self_rag": "Self-RAG",
    "08_ragas": "RAGAS",
    "09_instructgpt_rlhf": "InstructGPT / RLHF",
    "10_constitutional_ai": "Constitutional AI",
    "11_chain_of_thought_prompting": "Chain-of-Thought Prompting",
    "12_react": "ReAct",
    "13_hyde": "HyDE",
    "14_lost_in_the_middle": "Lost in the Middle",
    "15_mixtral_of_experts": "Mixtral of Experts",
    "16_mistral_7b": "Mistral 7B",
    "17_flare": "FLARE",
    "18_toolformer": "Toolformer",
    "19_sparks_of_agi_gpt4_technical_report": "Sparks of AGI (GPT-4)",
    "20_rag_survey": "RAG Survey",
}


def load_pdf(pdf_path: Path) -> list[Document]:
    """
    Load a single PDF and return one Document per page.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        List of Document objects, one per successfully extracted page.
        Pages that fail extraction are skipped and logged as warnings.

    Raises:
        FileNotFoundError: If the PDF does not exist on disk.
        PdfReadError: If the file is not a valid PDF.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            f"Download it from arXiv and place in data/papers/"
        )

    paper_id = pdf_path.stem
    title = PAPER_TITLES.get(paper_id, paper_id)

    logger.info(f"Loading: {title} ({pdf_path.name})")

    try:
        reader = PdfReader(str(pdf_path))
    except PdfReadError as e:
        logger.error(f"Cannot read PDF {pdf_path.name}: {e}")
        raise

    documents = []
    total_pages = len(reader.pages)

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text()

            # Skip pages with no extractable text (cover images, blank pages)
            if not text or len(text.strip()) < 50:
                logger.debug(
                    f"  Skipping page {page_num}/{total_pages} "
                    f"({pdf_path.name}) — insufficient text"
                )
                continue

            doc = Document(
                page_content=text.strip(),
                metadata={
                    "source": str(pdf_path),
                    "filename": pdf_path.name,
                    "paper_id": paper_id,
                    "title": title,
                    "page": page_num,
                    "total_pages": total_pages,
                },
            )
            documents.append(doc)

        except Exception as e:
            # Per-page error isolation — log and continue
            logger.warning(
                f"  Failed to extract page {page_num}/{total_pages} "
                f"from {pdf_path.name}: {e}"
            )
            continue

    logger.success(
        f"  Loaded {len(documents)}/{total_pages} pages from {title}"
    )
    return documents


def load_all_papers(
    papers_dir: Optional[Path] = None,
    paper_ids: Optional[list[str]] = None,
) -> list[Document]:
    """
    Load all PDFs from the papers directory.

    Args:
        papers_dir: Directory containing PDFs. Defaults to data/papers/.
        paper_ids: Optional list of paper_ids to load (e.g. ["01_attention..."]).
                   If None, loads all PDFs in the directory.

    Returns:
        List of all Document objects across all loaded papers.

    Example:
        # Load all 20 papers
        docs = load_all_papers()

        # Load only the first 3 papers for a quick smoke test
        docs = load_all_papers(paper_ids=[
            "01_attention_is_all_you_need",
            "02_bert",
            "03_gpt3"
        ])
    """
    target_dir = papers_dir or PAPERS_DIR

    if not target_dir.exists():
        raise FileNotFoundError(
            f"Papers directory not found: {target_dir}\n"
            f"Create data/papers/ and download the 20 PDFs from arXiv."
        )

    pdf_files = sorted(target_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {target_dir}\n"
            f"See README.md for the download table."
        )

    # Filter to requested paper_ids if specified
    if paper_ids:
        pdf_files = [
            p for p in pdf_files
            if p.stem in paper_ids
        ]
        if not pdf_files:
            raise ValueError(
                f"None of the requested paper_ids found in {target_dir}.\n"
                f"Requested: {paper_ids}"
            )

    logger.info(f"Starting ingestion of {len(pdf_files)} PDFs from {target_dir}")

    all_documents = []
    failed_files = []

    for pdf_path in pdf_files:
        try:
            docs = load_pdf(pdf_path)
            all_documents.extend(docs)
        except (FileNotFoundError, PdfReadError) as e:
            logger.error(f"Skipping {pdf_path.name}: {e}")
            failed_files.append(pdf_path.name)
            continue

    logger.info(
        f"Ingestion complete: {len(all_documents)} pages loaded "
        f"from {len(pdf_files) - len(failed_files)}/{len(pdf_files)} files"
    )

    if failed_files:
        logger.warning(f"Failed files: {failed_files}")

    return all_documents
