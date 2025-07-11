import os
import subprocess
import tempfile
from collections.abc import Iterator
from typing import Literal

import pdfplumber
from nemo_curator.datasets import DocumentDataset
from nemo_curator.download.doc_builder import (
    DocumentDownloader,
    DocumentExtractor,
    DocumentIterator,
    download_and_extract,
)
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir, get_all_files_paths_under


class S3FileDownloader(DocumentDownloader):
    def __init__(self, download_dir: str, aws: bool = False, verbose: bool = False):
        """
        Downloads files from S3 bucket
        
        Args:
            download_dir: Directory to store downloaded files
            aws: If True, uses s5cmd for downloading. If False, uses wget.
            verbose: If True, shows download command output
        """
        super().__init__()
        self._download_dir = download_dir
        self._aws = aws
        self._verbose = verbose

    def download(self, s3_path: str) -> str:
        """Download a file from S3 and return local path"""
        output_file = os.path.join(self._download_dir, os.path.basename(s3_path))
        
        if os.path.exists(output_file):
            print(f"File {output_file} exists. Not downloading")
            return output_file
            
        print(f"Downloading {s3_path} to {output_file}")
        
        if self._aws:
            cmd = ["s5cmd", "cp", s3_path, output_file]
        else:
            cmd = ["wget", s3_path, "-O", output_file]
            
        stdout = None if self._verbose else subprocess.DEVNULL
        stderr = None if self._verbose else subprocess.DEVNULL
        
        p = subprocess.run(cmd, stdout=stdout, stderr=stderr)
        if p.returncode != 0:
            print(f"Failed to download {s3_path} to {output_file}")
            
        return output_file


class S3FileIterator(DocumentIterator):
    def __init__(self, log_frequency: int = 1000):
        """
        Iterates through downloaded files
        
        Args:
            log_frequency: Log progress every N files
        """
        super().__init__()
        self._log_frequency = log_frequency
        self._counter = 0

    def iterate(self, file_path: str) -> Iterator[tuple[dict[str, str], str]]:
        """Yield file content with metadata"""
        self._counter = 0
        bname = os.path.basename(file_path)
        
        # For archives (like .tar.gz), extract and process each file
        if file_path.endswith(('.tar', '.tar.gz', '.tgz', '.zip')):
            with tempfile.TemporaryDirectory() as tmpdir:
                # Extract archive (implementation depends on archive type)
                # This is simplified - would need proper extraction logic
                extracted_files = get_all_files_paths_under(tmpdir)
                for item in extracted_files:
                    if self._counter > 0 and self._counter % self._log_frequency == 0:
                        print(f"Processed {self._counter} files from {file_path}")
                    self._counter += 1
                    
                    with open(item, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    yield {
                        "id": os.path.splitext(os.path.basename(item))[0],
                        "source_id": bname,
                        "file_type": os.path.splitext(item)[1][1:].lower(),
                    }, content
        else:
            # Single file processing
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            yield {
                "id": os.path.splitext(bname)[0],
                "source_id": bname,
                "file_type": os.path.splitext(file_path)[1][1:].lower(),
            }, content


class S3FileExtractor(DocumentExtractor):
    def __init__(self):
        """Extracts text from various file formats"""
        super().__init__()

    def extract(self, content: str) -> dict[str, str] | None:
        """
        Extract text from file content
        
        Args:
            content: File content to process
            
        Returns:
            Dictionary with extracted text or None if extraction failed
        """
        # For PDFs, we would use pdfplumber
        if isinstance(content, bytes):
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(content)
                    tmp.close()
                    with pdfplumber.open(tmp.name) as pdf:
                        text = "\n".join(page.extract_text() for page in pdf.pages)
                    os.unlink(tmp.name)
                    return {"text": text}
            except Exception:
                return None
                
        # For other text-based formats, return as-is
        if content and len(content.strip()) > 0:
            return {"text": content}
            
        return None


def download_s3_files(  # noqa: PLR0913
    output_path: str,
    s3_paths: list[str],
    output_type: Literal["jsonl", "parquet"] = "jsonl",
    aws: bool = False,
    raw_download_dir: str | None = None,
    keep_raw_download: bool = False,
    force_download: bool = False,
    url_limit: int | None = None,
    record_limit: int | None = None,
) -> DocumentDataset:
    """
    Download files from S3 and extract their text content
    
    Args:
        output_path: Root directory for output files
        s3_paths: List of S3 paths to download
        output_type: Output file format ("jsonl" or "parquet")
        aws: Use AWS s5cmd for downloads if True
        raw_download_dir: Directory for raw downloads
        keep_raw_download: Keep downloaded files after extraction
        force_download: Redownload even if output exists
        url_limit: Limit number of files to download
        record_limit: Limit records per file
        
    Returns:
        DocumentDataset with extracted content
    """
    if url_limit:
        s3_paths = s3_paths[:url_limit]
        
    output_paths = [os.path.join(output_path, f"{os.path.basename(url)}.{output_type}") 
                   for url in s3_paths]

    if not raw_download_dir:
        raw_download_dir = os.path.join(output_path, "downloads")
    expand_outdir_and_mkdir(raw_download_dir)
    
    downloader = S3FileDownloader(raw_download_dir, aws=aws)
    iterator = S3FileIterator()
    extractor = S3FileExtractor()

    output_format = {
        "text": str,
        "id": str,
        "source_id": str,
        "file_type": str,
        "file_name": str,
    }

    return download_and_extract(
        s3_paths,
        output_paths,
        downloader,
        iterator,
        extractor,
        output_format,
        output_type=output_type,
        keep_raw_download=keep_raw_download,
        force_download=force_download,
        filename_col="file_name",
        record_limit=record_limit,
    )
