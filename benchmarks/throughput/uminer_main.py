import time
import torch

import click
import pypdfium2 as pdfium
from tqdm import tqdm

import tracemalloc

from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.config.make_content_config import MakeMode, DropMode
from magic_pdf.operators.models import InferenceResult, PipeResult
from magic_pdf.data.data_reader_writer import FileBasedDataReader
        
@click.command(help="Benchmark PDF to MD conversion throughput.")
@click.argument("pdf_path", type=str)
@click.option("--trace_memory", type=bool, help="Trace memory usage", default=False)
@click.option("--loops", type=int, help="Number of benchmark loops", default=10)
def main(pdf_path: str, trace_memory: bool, loops: int):
    print(f"Converting {pdf_path} to markdown...")
    pdf = pdfium.PdfDocument(pdf_path)
    page_count = len(pdf)
    pdf.close()

    torch.cuda.reset_peak_memory_stats()
    times = []
    if trace_memory:
        print("Tracing memory")
        tracemalloc.start()
        loops = 1

    for i in tqdm(range(loops), desc="Benchmarking"):
        start = time.time()
        reader1 = FileBasedDataReader("")
        pdf_bytes = reader1.read(pdf_path)
        ds = PymuDocDataset(pdf_bytes)
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result: InferenceResult = ds.apply(doc_analyze, ocr=True)
            pipe_result: PipeResult = infer_result.pipe_ocr_mode(imageWriter=None)
        else:
            infer_result: InferenceResult = ds.apply(doc_analyze, ocr=False)
            pipe_result: PipeResult = infer_result.pipe_txt_mode(imageWriter=None)

        pipe_result.get_markdown('uminer_images', drop_mode=DropMode.SINGLE_PAGE, md_make_mode=MakeMode.MM_MD)

        total = time.time() - start
        times.append(total)

    print(f"Converted {page_count} pages in {sum(times)/len(times):.2f} seconds.")
    
    max_gpu_vram = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"Max GPU VRAM: {max_gpu_vram:.2f} GB")
    
    if trace_memory:
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage: {current / 10**9} GB")
        print(f"Peak memory usage: {peak / 10**9} GB")
        tracemalloc.stop()

if __name__ == "__main__":
    main()