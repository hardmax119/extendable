# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import tempfile
from collections.abc import Iterator
from typing import Literal, List
from urllib.parse import urlparse

import boto3
import fitz  # PyMuPDF
import markdown
from bs4 import BeautifulSoup

from nemo_curator.datasets import DocumentDataset
from nemo_curator.download.doc_builder import (
    DocumentDownloader,
    DocumentExtractor,
    DocumentIterator,
    download_and_extract,
)
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir


class S3DocumentDownloader(DocumentDownloader):
    """
    Downloads documents from an S3 bucket
    """

    def __init__(self, download_dir: str, aws_profile: str = None, verbose: bool = False):
        """
        Creates a downloader for S3 documents

        Args:
          download_dir: Path to store raw downloaded files
          aws_profile: AWS profile name to use for authentication. If None, uses default credentials.
          verbose: If True, logs stdout and stderr of the download command
        """
        super().__init__()
        self._download_dir = download_dir
        self._aws_profile = aws_profile
        self._verbose = verbose
        
        # Initialize S3 client
        session = boto3.Session(profile_name=self._aws_profile)
        self._s3_client = session.client('s3')

    def download(self, url: str) -> str:
        """
        Downloads a file from S3

        Args:
          url: S3 URL in the format s3://bucket-name/path/to/file
          
        Returns:
          Local path to the downloaded file
        """
        # Parse the S3 URL
        parsed_url = urlparse(url)
        if parsed_url.scheme != 's3':
            raise ValueError(f"URL must be an S3 URL (s3://), got {url}")
        
        bucket = parsed_url.netloc
        key = parsed_url.path.lstrip('/')
        
        # Create output filename based on the key
        filename = os.path.basename(key)
        output_file = os.path.join(self._download_dir, filename)
        
        # Check if file already exists
        if os.path.exists(output_file):
            print(f"File: {output_file} exists. Not downloading")
            return output_file
        
        # Download the file
        print(f"Downloading {url} and writing to {output_file}")
        try:
            if self._verbose:
                print(f"Downloading from bucket: {bucket}, key: {key}")
            
            # Use s5cmd for efficiency if available
            try:
                cmd = ["s5cmd", "cp", url, output_file]
                if self._verbose:
                    stdout, stderr = None, None
                else:
                    stdout, stderr = subprocess.DEVNULL, subprocess.DEVNULL
                p = subprocess.run(
                    cmd,
                    stdout=stdout,
                    stderr=stderr,
                )
                if p.returncode == 0:
                    return output_file
                else:
                    print(f"s5cmd failed, falling back to boto3")
            except FileNotFoundError:
                # s5cmd not available, continue with boto3
                pass
                
            # Fallback to boto3
            self._s3_client.download_file(bucket, key, output_file)
            return output_file
            
        except Exception as e:
            print(f"Failed to download {url} to {output_file}: {e}")
            # Return the URL so the iterator can handle the failure gracefully
            return url


class S3DocumentIterator(DocumentIterator):
    """
    Iterator for documents downloaded from S3
    """
    
    def __init__(self, log_frequency: int = 100):
        super().__init__()
        self._counter = 0
        self._log_frequency = log_frequency
        
    def iterate(self, file_path: str) -> Iterator[tuple[dict[str, str], str]]:
        """
        Iterates over a document, yielding metadata and content

        Args:
          file_path: Path to the downloaded file
          
        Yields:
          Tuple of (metadata, content)
        """
        self._counter = 0
        
        # Get file extension and source information
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # Read the file 
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                
            self._counter += 1
            if self._counter % self._log_frequency == 0:
                print(f"Processed {self._counter} files")
                
            # Create metadata
            meta = {
                "file_name": file_name,
                "file_type": file_ext[1:] if file_ext.startswith('.') else file_ext,
                "source_id": file_name,
            }
            
            yield meta, content
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")


class S3DocumentExtractor(DocumentExtractor):
    """
    Base extractor for documents downloaded from S3
    """
    
    def __init__(self):
        super().__init__()
    
    def extract(self, content: bytes) -> dict[str, str] | None:
        """
        Extracts text from document content based on file type
        
        Args:
          content: Raw content of the document
          
        Returns:
          Dictionary with extracted text or None if extraction failed
        """
        return None


class PDFExtractor(S3DocumentExtractor):
    """
    Extractor for PDF documents
    """
    
    def extract(self, content: bytes) -> dict[str, str] | None:
        """
        Extracts text from PDF content
        
        Args:
          content: Raw PDF content
          
        Returns:
          Dictionary with extracted text or None if extraction failed
        """
        try:
            # Create a temporary file to write the PDF content
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            
            # Extract text using PyMuPDF
            text = ""
            with fitz.open(temp_path) as doc:
                for page in doc:
                    text += page.get_text()
                    text += "\n\n"  # Add spacing between pages
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            if text.strip():
                return {"text": text.strip()}
            return None
            
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return None


class TXTExtractor(S3DocumentExtractor):
    """
    Extractor for plain text documents
    """
    
    def extract(self, content: bytes) -> dict[str, str] | None:
        """
        Extracts text from plain text content
        
        Args:
          content: Raw text content
          
        Returns:
          Dictionary with extracted text or None if extraction failed
        """
        try:
            # Try to decode with utf-8 first
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                # Try to detect encoding
                from charset_normalizer import detect
                detected_encoding = detect(content)["encoding"]
                if not detected_encoding:
                    return None
                text = content.decode(detected_encoding)
            
            if text.strip():
                return {"text": text.strip()}
            return None
            
        except Exception as e:
            print(f"Error extracting text from TXT: {e}")
            return None


class HTMLExtractor(S3DocumentExtractor):
    """
    Extractor for HTML documents
    """
    
    def extract(self, content: bytes) -> dict[str, str] | None:
        """
        Extracts text from HTML content
        
        Args:
          content: Raw HTML content
          
        Returns:
          Dictionary with extracted text or None if extraction failed
        """
        try:
            # Try to decode with utf-8 first
            try:
                html = content.decode('utf-8')
            except UnicodeDecodeError:
                # Try to detect encoding
                from charset_normalizer import detect
                detected_encoding = detect(content)["encoding"]
                if not detected_encoding:
                    return None
                html = content.decode(detected_encoding)
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            text = soup.get_text(separator='\n')
            
            # Clean up text - remove multiple newlines and whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            if text.strip():
                return {"text": text.strip()}
            return None
            
        except Exception as e:
            print(f"Error extracting text from HTML: {e}")
            return None


class MarkdownExtractor(S3DocumentExtractor):
    """
    Extractor for Markdown documents
    """
    
    def extract(self, content: bytes) -> dict[str, str] | None:
        """
        Extracts text from Markdown content
        
        Args:
          content: Raw Markdown content
          
        Returns:
          Dictionary with extracted text or None if extraction failed
        """
        try:
            # Try to decode with utf-8 first
            try:
                md_text = content.decode('utf-8')
            except UnicodeDecodeError:
                # Try to detect encoding
                from charset_normalizer import detect
                detected_encoding = detect(content)["encoding"]
                if not detected_encoding:
                    return None
                md_text = content.decode(detected_encoding)
            
            # Convert markdown to HTML
            html = markdown.markdown(md_text)
            
            # Parse HTML with BeautifulSoup to extract clean text
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text(separator='\n')
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)
            
            if text.strip():
                return {"text": text.strip()}
            return None
            
        except Exception as e:
            print(f"Error extracting text from Markdown: {e}")
            return None


class CompositeS3DocumentExtractor(S3DocumentExtractor):
    """
    Composite extractor that selects the appropriate extractor based on file type
    """
    
    def __init__(self):
        super().__init__()
        self.extractors = {
            'pdf': PDFExtractor(),
            'txt': TXTExtractor(),
            'html': HTMLExtractor(),
            'htm': HTMLExtractor(),
            'md': MarkdownExtractor(),
        }
    
    def extract(self, content: bytes) -> dict[str, str] | None:
        """
        Extracts text based on file type metadata
        
        Args:
          content: Tuple of (metadata, raw_content)
          
        Returns:
          Dictionary with extracted text or None if extraction failed
        """
        # This method will be overridden in the download_s3_documents function
        # to properly handle the metadata and content
        return None


def download_s3_documents(
    output_path: str,
    s3_urls: List[str],
    output_type: Literal["jsonl", "parquet"] = "jsonl",
    aws_profile: str = None,
    raw_download_dir: str = None,
    keep_raw_download: bool = False,
    force_download: bool = False,
    url_limit: int = None,
    record_limit: int = None,
) -> DocumentDataset:
    """
    Downloads documents from S3 and extracts their text content.
    
    Args:
        output_path (str): The root directory for managing download and extraction.
        s3_urls (List[str]): List of S3 URLs to download (s3://bucket-name/path/to/file).
        output_type (Literal["jsonl", "parquet"]): The file format for the extracted output.
            Default is "jsonl".
        aws_profile (str, optional): AWS profile name to use for authentication.
            If None, uses default credentials.
        raw_download_dir (str, optional): Directory to temporarily store raw files.
            If not provided, defaults to a "downloads" folder within output_path.
        keep_raw_download (bool): If True, retains the downloaded raw files after extraction.
            Default is False.
        force_download (bool): If False, skips re-downloading or re-extracting if outputs already exist.
            Default is False.
        url_limit (int, optional): Maximum number of URLs to download.
            If None, all provided URLs are downloaded.
        record_limit (int, optional): Maximum number of records to extract from each file.
            If None, all available records are extracted.
            
    Returns:
        DocumentDataset: A dataset object containing the extracted documents.
    """
    if url_limit:
        s3_urls = s3_urls[:url_limit]
    
    # Create output paths for each URL
    output_paths = [os.path.join(output_path, os.path.basename(url) + f".{output_type}") for url in s3_urls]
    
    # Set up download directory
    if not raw_download_dir:
        raw_download_dir = os.path.join(output_path, "downloads")
    expand_outdir_and_mkdir(raw_download_dir)
    
    # Initialize components
    downloader = S3DocumentDownloader(raw_download_dir, aws_profile=aws_profile)
    iterator = S3DocumentIterator()
    
    # Create a special extractor that selects the appropriate extractor based on file type
    class FileTypeAwareExtractor(CompositeS3DocumentExtractor):
        def extract(self, content_tuple: tuple[dict[str, str], bytes]) -> dict[str, str] | None:
            metadata, content = content_tuple
            file_type = metadata.get("file_type", "")
            
            # Select the appropriate extractor based on file type
            extractor = self.extractors.get(file_type.lower())
            if extractor:
                extracted = extractor.extract(content)
                if extracted:
                    # Add metadata to the extraction result
                    return {**extracted, **metadata}
            return None
    
    extractor = FileTypeAwareExtractor()
    
    # Define output format
    output_format = {
        "text": str,
        "file_name": str,
        "source_id": str,
        "file_type": str,
        "file_name": str,
    }
    return download_and_extract(
        s3_urls,
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
