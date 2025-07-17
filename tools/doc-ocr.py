from collections.abc import Generator
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from PyPDF2 import PdfReader, PdfWriter
from enum import Enum
import requests
import mdformat
import time
import base64
import io

class PaddlepaddleDocOcrTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        file_objs = tool_parameters.get("p_file")
        segmentation = tool_parameters.get("p_segmentation", None)
        return_type = tool_parameters.get("return_type")
        
        if not file_objs:
            raise ValueError("No files provided for OCR processing.")
        
        # 文件层多线程，最多4线程
        with ThreadPoolExecutor(max_workers=min(4, len(file_objs))) as executor:
            # 提交所有文件处理任务，保持顺序
            future_to_serial = {}
            for serial, file_obj in enumerate(file_objs):
                future = executor.submit(self._process_file, serial, file_obj, segmentation)
                future_to_serial[future] = serial
            
            # 收集结果并按序号排序
            results = {}
            for future in as_completed(future_to_serial):
                try:
                    result = future.result()
                    results.update(result)
                except Exception as e:
                    serial = future_to_serial[future]
                    results[serial] = f"Error processing file {serial}: {str(e)}"
            
            # 按顺序拼接结果
            final_result = ""
            for i in sorted(results.keys()):
                final_result += results[i] + "\n\n"
        
        return_res = self.create_json_message({
            "result": final_result.strip()
        }) if return_type == "text" else self.create_blob_message(
            blob=final_result.encode('utf-8'),
            meta={
                "filename": "ocr_result.md",
                "content_type": "text/plain"
            }
        )

        yield return_res

    def _process_file(self, serial: int, file_obj: Any, segmentation: int| None) -> dict[int, str]:
        extension = file_obj.extension.lower()
        if not (file_obj.url):
            raise ValueError("Input file does not contain a valid URL.")
        
        parsed = ""
        if (extension == ".pdf"): 
            parsed = mdformat.text(self._process_pdf(file_obj, segmentation))
        elif (extension in [".jpg", ".jpeg", ".png", ".bmp"]):
            parsed = mdformat.text(self._process_image(file_obj))
        else:
            raise ValueError(f"Unsupported file type: {extension}")
        
        return {serial: parsed}

    #region 处理方法入口
    def _process_image(self, file_obj: Any)-> str:
        uri: Any = file_obj.url
        
        image: bytes = self._get_file_by_url(uri)
        if not image:
            raise ValueError("Failed to retrieve image from the provided URL.")

        return self._ocr(image, self.OcrType.IMAGE)
    

    def _process_pdf(self, file_obj: Any, segmentation: int | None) -> str:
        uri: Any = file_obj.url
        
        pdf_file: bytes = self._get_file_by_url(uri)
        pdf_slices: list[bytes] = []

        if segmentation:
            pdf_slices = self._split_pdf(pdf_file, segmentation)
        else:
            pdf_slices.append(pdf_file)

        # 最多3线程多线程
        if len(pdf_slices) == 1:
            ocr_result = self._ocr(pdf_slices[0], self.OcrType.PDF)
            return ocr_result
        else:
            with ThreadPoolExecutor(max_workers=min(3, len(pdf_slices))) as executor:
                # 提交所有分片处理任务，保持顺序
                future_to_index = {}
                for index, pdf_slice in enumerate(pdf_slices):
                    future = executor.submit(self._ocr, pdf_slice, self.OcrType.PDF)
                    future_to_index[future] = index
                
                # 收集结果并按索引排序
                slice_results = {}
                for future in as_completed(future_to_index):
                    try:
                        result = future.result()
                        index = future_to_index[future]
                        slice_results[index] = result
                    except Exception as e:
                        index = future_to_index[future]
                        slice_results[index] = f"Error processing PDF slice {index}: {str(e)}"
                
                # 按顺序拼接分片结果
                final_result = ""
                for i in sorted(slice_results.keys()):
                    final_result += slice_results[i] + "\n\n"
                
                return final_result.strip()
    #endregion


    #region 辅助方法
    class OcrType(Enum):
        IMAGE = 1
        PDF = 0

    def _ocr(self, file: bytes, type: OcrType) -> str:
        ocr_url = self.runtime.credentials.get("ocr_url", "")
        if not ocr_url:
            raise ValueError("OCR URL is not configured in the credentials.")
        
        payload = {
            "file": base64.b64encode(file).decode('ascii'),
            "fileType": type.value
        }
        response = requests.post(ocr_url, json=payload)

        if response.status_code != 200:
            raise ValueError(f"Failed to process OCR: {response.text}")
        
        data = response.json()
        if "result" not in data or "layoutParsingResults" not in data["result"]:
            raise ValueError("OCR response does not contain expected fields.")
        results = data["result"]["layoutParsingResults"]
        if not results or "markdown" not in results[0] or "text" not in results[0]["markdown"]:
            raise ValueError("OCR response does not contain markdown text.")
        
        return results[0]["markdown"]["text"]


    def _get_file_by_url(self, uri: Any) -> bytes:
        base_url = self.runtime.credentials.get("base_url", "http://nginx:80")
        for attempt in range(3):
            response = requests.get(base_url + uri)
            if response.status_code == 200:
                return response.content
            if attempt < 2:
                time.sleep(1)
        raise ValueError(f"Failed to fetch file from URL after 3 attempts: {base_url}{uri}")

    
    def _split_pdf(self, file: bytes, segmentation: int) -> list[bytes]:
        reader = PdfReader(io.BytesIO(file))
        total_pages = len(reader.pages)
        slices = []

        for start in range(0, total_pages, segmentation):
            writer = PdfWriter()

            for i in range(start, min(start + segmentation, total_pages)):
                writer.add_page(reader.pages[i])

            output_stream = io.BytesIO()
            writer.write(output_stream)
            slices.append(output_stream.getvalue())
            output_stream.close()

        return slices
    #endregion
