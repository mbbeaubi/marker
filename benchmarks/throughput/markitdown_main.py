import time
import torch

import click
import pypdfium2 as pdfium
from tqdm import tqdm

import tracemalloc

from markitdown import MarkItDown
        
@click.command(help="Benchmark PDF to MD conversion throughput.")
@click.argument("pdf_path", type=str)
@click.option("--trace_memory", type=bool, help="Trace memory usage", default=False)
@click.option("--loops", type=int, help="Number of benchmark loops", default=10)
def main(pdf_path: str, trace_memory: bool, loops: int):
    print(f"Converting {pdf_path} to markdown...")
    pdf = pdfium.PdfDocument(pdf_path)
    page_count = len(pdf)
    pdf.close()


    times = []
    if trace_memory:
        print("Tracing memory")
        tracemalloc.start()
        loops = 1

    md = MarkItDown(enable_plugins=False) # Set to True to enable plugins

    for i in tqdm(range(loops), desc="Benchmarking"):
        start = time.time()
        md.convert(pdf_path).text_content
        total = time.time() - start
        times.append(total)

    print(f"Converted {page_count} pages in {sum(times)/len(times):.2f} seconds.")
    
    if trace_memory:
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage: {current / 10**9} GB")
        print(f"Peak memory usage: {peak / 10**9} GB")
        tracemalloc.stop()

if __name__ == "__main__":
    main()