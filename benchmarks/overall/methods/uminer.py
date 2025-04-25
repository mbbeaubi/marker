import tempfile
import time
import tika
import os

from benchmarks.overall.methods import BaseMethod, BenchmarkResult

class UminerMethod(BaseMethod):
    model_dict: dict = None
    use_llm: bool = False

    def __call__(self, sample) -> BenchmarkResult:
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
        from magic_pdf.config.enums import SupportedPdfParseMethod
        from magic_pdf.config.make_content_config import MakeMode, DropMode
        from magic_pdf.operators.models import InferenceResult, PipeResult

        pdf_bytes = sample["pdf"]  # This is a single page PDF
        ds = PymuDocDataset(pdf_bytes)

        start = time.time()
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result: InferenceResult = ds.apply(doc_analyze, ocr=True)
            pipe_result: PipeResult = infer_result.pipe_ocr_mode(imageWriter=None)
        else:
            infer_result: InferenceResult = ds.apply(doc_analyze, ocr=False)
            pipe_result: PipeResult = infer_result.pipe_txt_mode(imageWriter=None)

        md_text = pipe_result.get_markdown('uminer_images', drop_mode=DropMode.SINGLE_PAGE, md_make_mode=MakeMode.MM_MD)
        total = time.time() - start

        return {
            "markdown": md_text,
            "time": total
        }

