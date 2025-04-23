import tempfile
import time
import tika
import os

from benchmarks.overall.methods import BaseMethod, BenchmarkResult

class TikaMethod(BaseMethod):
    model_dict: dict = None
    use_llm: bool = False
    server_started: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not TikaMethod.server_started:
            tika.initVM()
            TikaMethod.server_started = True

        self.timeout = int(os.getenv("TIKA_TIMEOUT", "300"))
        self.skip_ocr = os.getenv("TIKA_SKIP_OCR", "true")

    def __call__(self, sample) -> BenchmarkResult:
        from tika import parser
        from markdownify import markdownify as md

        pdf_bytes = sample["pdf"]  # This is a single page PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", mode="wb") as f:
            f.write(pdf_bytes)
            start = time.time()
            parsed = parser.from_file(f.name, requestOptions={"timeout": self.timeout}, headers={'X-Tika-OCRskipOcr': self.skip_ocr.lower()})
            md_text = md(parsed['content'])
            total = time.time() - start

        return {
            "markdown": md_text,
            "time": total
        }

