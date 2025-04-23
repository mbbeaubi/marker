import tempfile
import time

from benchmarks.overall.methods import BaseMethod, BenchmarkResult

from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf

class UnstructuredMethod(BaseMethod):
    model_dict: dict = None
    use_llm: bool = False

    def __call__(self, sample) -> BenchmarkResult:
        pdf_bytes = sample["pdf"]  # This is a single page PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", mode="wb") as f:
            f.write(pdf_bytes)
            start = time.time()
            elements = partition_pdf(filename=f.name, strategy='hi_res', hi_res_model_name='yolox', infer_table_structure=True)
            md_text = elements_to_md(elements)
            total = time.time() - start

        return {
            "markdown": md_text,
            "time": total
        }
    
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, Title, ListItem, NarrativeText, Element
from markdownify import markdownify as md

def elements_to_md(elements: list[Element]):
    """
    Convert unstructured.io Element objects to a single HTML string.
    tables keep their inferred structure.
    """
    html_parts = ["<html><head><meta charset='utf-8'></head><body>"]
    list_started = False
    for el in elements:
        if list_started and not isinstance(el, ListItem):
            html_parts.append("</ul>")
            list_started = False
        
        if isinstance(el, Title):
            #level = el.metadata.heading_level or 1
            html_parts.append(f"<h1>{el.text}</h1>")

        elif isinstance(el, ListItem):
            if not list_started:
                html_parts.append("<ul>")
                list_started = True
            html_parts.append(f"<li>{el.text}</li>")

        elif isinstance(el, Table):
            html_parts.append(el.metadata.text_as_html)

        elif isinstance(el, NarrativeText):
            html_parts.append(f"<p>{el.text}</p>")

        # add other element types as needed …

    # close an open <ul> if the last element was a ListItem
    if list_started:
        html_parts.append("</ul>")

    html_parts.append("</body></html>")
    html_str = "\n".join(html_parts)
    return md(html_str)