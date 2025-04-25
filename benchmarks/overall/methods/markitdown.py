import tempfile
import time
import tika
import os

from benchmarks.overall.methods import BaseMethod, BenchmarkResult

class MarkitdownMethod(BaseMethod):
    model_dict: dict = None
    use_llm: bool = False

    def __call__(self, sample) -> BenchmarkResult:

        from markitdown import MarkItDown
        md = MarkItDown(enable_plugins=False) # Set to True to enable plugins
        
        pdf_bytes = sample["pdf"]  # This is a single page PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", mode="wb") as f:
            f.write(pdf_bytes)
            start = time.time()
            md_text = md.convert(f.name).text_content
            total = time.time() - start

        return {
            "markdown": md_text,
            "time": total
        }

