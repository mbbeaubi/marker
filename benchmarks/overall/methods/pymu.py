import tempfile
import time

from benchmarks.overall.methods import BaseMethod, BenchmarkResult

class PyMuMethod(BaseMethod):
    model_dict: dict = None
    use_llm: bool = False

    def __call__(self, sample) -> BenchmarkResult:
        import pymupdf
        from pymupdf4llm.helpers.pymupdf_rag import to_markdown

        pdf_bytes = sample["pdf"]  # This is a single page PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", mode="wb") as f:
            f.write(pdf_bytes)
            start = time.time()
            doc = pymupdf.open(f.name)
            md_text = to_markdown(doc)
            total = time.time() - start

        return {
            "markdown": md_text,
            "time": total
        }

